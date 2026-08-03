"""GCS I/O helpers for reading input variables and writing annual metrics."""

from __future__ import annotations

from datetime import datetime

import gcsfs
import xarray as xr

from config import (
    GCS_INPUT_TEMPLATE,
    GCS_OUTPUT_TEMPLATE,
    GCS_WL_STATS_TEMPLATE,
    MODEL,
    OUT_CHUNKS,
    SCENARIO,
    VARIABLE_SOURCES,
    WL_OUT_CHUNKS,
)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# Unit conversions applied on read so all metric functions receive
# temperatures in °C and precipitation in mm/day.
_UNIT_CONVERSIONS: dict[str, callable] = {
    "tasmax": lambda da: da - 273.15,
    "tasmin": lambda da: da - 273.15,
    "tas": lambda da: da - 273.15,
    "pr": lambda da: da * 86400.0,
}


def open_variable(variable: str) -> xr.DataArray:
    """Open a climate variable from GCS as a lazy dask-backed DataArray.

    The variable's source location and chunking are looked up in
    VARIABLE_SOURCES, so this works for both daily variables (tasmax, tasmin,
    tas, wetbulbmax, pr) and monthly ones (wb_percentile).

    Temperatures (tasmax, tasmin, tas) are converted from K to °C.
    Precipitation (pr) is converted from kg m⁻² s⁻¹ to mm/day.
    """
    src = VARIABLE_SOURCES[variable]
    path = GCS_INPUT_TEMPLATE.format(
        dir=src["dir"],
        prefix=src["prefix"],
        model=MODEL,
        scenario=SCENARIO,
        freq=src["freq"],
    )
    fs = gcsfs.GCSFileSystem()
    da = xr.open_zarr(fs.get_mapper(path), chunks=src["chunks"], consolidated=False)[
        variable
    ]
    if variable in _UNIT_CONVERSIONS:
        da = _UNIT_CONVERSIONS[variable](da)
    return da


def open_metric(metric: str) -> xr.DataArray:
    """Open a previously-computed annual metric from GCS as a lazy DataArray.

    Symmetric to `open_variable()`, but reads back an annual metric written by
    `save_metric()` (via `GCS_OUTPUT_TEMPLATE`) rather than a raw daily input
    variable.
    """
    path = GCS_OUTPUT_TEMPLATE.format(metric=metric, model=MODEL, scenario=SCENARIO)
    fs = gcsfs.GCSFileSystem()
    var_name = metric.replace("-", "_")
    da = xr.open_zarr(fs.get_mapper(path), chunks=OUT_CHUNKS, consolidated=False)[
        var_name
    ]

    # Drop non-dimension coordinates (e.g. a scalar 'height' coordinate
    # carried over from the source tasmax/tasmin/tas/... input) so they don't
    # propagate into downstream outputs derived from this DataArray, such as
    # the warming-level stats written by save_wl_stats().
    extra_coords = [c for c in da.coords if c not in da.dims]
    if extra_coords:
        da = da.drop_vars(extra_coords)

    return da


def save_metric(da: xr.DataArray, metric: str) -> None:
    """Write an annual metric DataArray to GCS as a zarr store.

    Writes incrementally in batches along `time`, sized to
    `OUT_CHUNKS["time"]`, rather than triggering one dask computation over
    the entire (potentially multi-decade) record. A single `.to_zarr()` call
    over the whole record hands dask a graph where many chunks' worth of
    already-computed-but-not-yet-written results can accumulate in worker
    memory/spill before anything is flushed to durable storage -- with a
    limited local disk budget for spilling, this can exhaust it even when
    each individual task is cheap. Writing batch-by-batch bounds how much
    unwritten data can be resident at once to roughly one batch's worth,
    since each batch is fully computed and flushed before the next begins.

    Trade-off: this sacrifices the all-or-nothing atomicity of a single
    `to_zarr()` call -- if the job is interrupted mid-run, the output store
    will contain a partial result (leading batches written, the rest
    missing) rather than being complete or untouched.

    The variable is named using the metric name with hyphens replaced by
    underscores.
    """
    path = GCS_OUTPUT_TEMPLATE.format(metric=metric, model=MODEL, scenario=SCENARIO)
    fs = gcsfs.GCSFileSystem()
    var_name = metric.replace("-", "_")

    da = da.chunk(OUT_CHUNKS)
    batch_size = OUT_CHUNKS["time"]
    n_time = da.sizes["time"]
    n_batches = -(-n_time // batch_size)  # ceil division

    for i, start in enumerate(range(0, n_time, batch_size), 1):
        batch = da.isel(time=slice(start, start + batch_size)).to_dataset(name=var_name)
        if start == 0:
            batch.to_zarr(
                fs.get_mapper(path), mode="w", zarr_format=3, consolidated=False
            )
        else:
            batch.to_zarr(
                fs.get_mapper(path),
                mode="a",
                append_dim="time",
                zarr_format=3,
                consolidated=False,
            )
        _log(f"  wrote batch {i}/{n_batches}")


def save_wl_stats(da: xr.DataArray, diff_da: xr.DataArray, metric: str) -> None:
    """Write per-warming-level statistics and their diff-from-baseline to GCS.

    `da` and `diff_da` are expected to have matching dims (wl, stat,
    latitude, longitude) and share the same `wl` coordinate -- `diff_da` is
    derived from `da` (see `wl_stats.compute_wl_diff()`). Both are written
    as separate data variables in the same zarr store. As with
    `save_metric()`, this writes incrementally rather than in one `to_zarr()`
    call -- here, one warming level at a time via `append_dim="wl"` -- to
    bound how much computed-but-unwritten data can accumulate in worker
    memory/spill before being flushed. Each warming level's stats depend on a
    full 21-year window of the (spatially-chunked) annual metric, so, as with
    `save_metric()`, this is not atomic: an interrupted run leaves a partial
    output store (leading warming levels written, the rest missing).

    The main variable is named using the metric name with hyphens replaced
    by underscores; the diff variable is that name prefixed with `diff_`.
    """
    path = GCS_WL_STATS_TEMPLATE.format(metric=metric, model=MODEL, scenario=SCENARIO)
    fs = gcsfs.GCSFileSystem()
    var_name = metric.replace("-", "_")
    diff_var_name = f"diff_{var_name}"

    da = da.chunk(WL_OUT_CHUNKS)
    diff_da = diff_da.chunk(WL_OUT_CHUNKS)
    n_wl = da.sizes["wl"]

    for i in range(n_wl):
        batch = xr.Dataset(
            {
                var_name: da.isel(wl=slice(i, i + 1)),
                diff_var_name: diff_da.isel(wl=slice(i, i + 1)),
            }
        )
        if i == 0:
            batch.to_zarr(
                fs.get_mapper(path), mode="w", zarr_format=3, consolidated=False
            )
        else:
            batch.to_zarr(
                fs.get_mapper(path),
                mode="a",
                append_dim="wl",
                zarr_format=3,
                consolidated=False,
            )
        _log(f"  wrote wl batch {i + 1}/{n_wl}")
