"""GCS I/O helpers for reading input variables and writing annual metrics."""

from __future__ import annotations

from datetime import datetime

import gcsfs
import xarray as xr

from config import (
    CHUNKS,
    GCS_INPUT_TEMPLATE,
    GCS_OUTPUT_TEMPLATE,
    MODEL,
    OUT_CHUNKS,
    SCENARIO,
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
    """Open a daily climate variable from GCS as a lazy dask-backed DataArray.

    Temperatures (tasmax, tasmin, tas) are converted from K to °C.
    Precipitation (pr) is converted from kg m⁻² s⁻¹ to mm/day.
    """
    path = GCS_INPUT_TEMPLATE.format(var=variable, model=MODEL, scenario=SCENARIO)
    fs = gcsfs.GCSFileSystem()
    da = xr.open_zarr(fs.get_mapper(path), chunks=CHUNKS, consolidated=False)[variable]
    if variable in _UNIT_CONVERSIONS:
        da = _UNIT_CONVERSIONS[variable](da)
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
