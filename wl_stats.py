"""Per-warming-level statistics over 21-year windows of an annual metric.

For a given annual climate metric (already computed by `metrics.py`/`run.py`),
slices its time series into 21-year windows centered on each warming level's
year (from a wls_yrs CSV), and computes a small set of summary statistics
(order statistics via the nearest-rank method, plus the mean) within each
window. The result is a lazy DataArray with two new dimensions: `wl`
(warming level) and `stat` (statistic label).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from config import QUANTILES, STAT_LABELS, WL_WINDOW_HALF_WIDTH


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_warming_levels(path: str) -> pd.DataFrame:
    """Read a wls_yrs CSV (columns: wl, yr) into a DataFrame."""
    return pd.read_csv(path)


def _nearest_rank_quantile(x: np.ndarray, p: float) -> float:
    """Nearest-rank quantile: rank = ceil(p * n), clamped to [1, n].

    Unlike interpolation-based quantile methods, this always returns an
    actual observed value (e.g. for n=21, p=0.05: rank = ceil(1.05) = 2, so
    the result is exactly the 2nd-smallest of the 21 values).
    """
    valid = x[~np.isnan(x)]
    n = len(valid)
    if n == 0:
        return np.nan
    rank = int(np.ceil(p * n))
    rank = min(max(rank, 1), n)
    return np.partition(valid, rank - 1)[rank - 1]


def _window_stats(da_window: xr.DataArray) -> xr.DataArray:
    """Compute quantiles (nearest-rank) + mean over a window's `time` dim.

    `da_window` should already be the sliced 21-year window. Its `time` dim
    is rechunked to a single chunk (safe here: only ~21 timesteps per
    spatial tile) before applying the per-pixel numpy function, following
    the same one-window-at-a-time pattern used for `_top_n_mean` in
    metrics.py.
    """
    fractions = list(QUANTILES.values())

    def _fn(x: np.ndarray) -> np.ndarray:
        quantile_vals = [_nearest_rank_quantile(x, p) for p in fractions]
        valid = x[~np.isnan(x)]
        mean_val = np.mean(valid) if len(valid) else np.nan
        return np.array(quantile_vals + [mean_val], dtype=float)

    da_window = da_window.chunk({"time": -1})
    result = xr.apply_ufunc(
        _fn,
        da_window,
        input_core_dims=[["time"]],
        output_core_dims=[["stat"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"stat": len(STAT_LABELS)}},
    )
    result = result.assign_coords(stat=STAT_LABELS)
    return result


def compute_wl_stats(da: xr.DataArray, wl_table: pd.DataFrame) -> xr.DataArray:
    """Compute stats over a 21-year window per warming level.

    For each (wl, yr) row in `wl_table`, checks whether the window
    [yr - WL_WINDOW_HALF_WIDTH, yr + WL_WINDOW_HALF_WIDTH] is fully covered
    by `da`'s available years. Warming levels whose window isn't fully
    covered are skipped with a logged warning. Returns a combined, still-lazy
    DataArray with dims (wl, stat, latitude, longitude).
    """
    available_years = da.time.dt.year.values
    year_min, year_max = int(available_years.min()), int(available_years.max())

    per_wl_results = []
    for _, row in wl_table.iterrows():
        wl = row["wl"]
        yr = int(row["yr"])
        start_yr, end_yr = yr - WL_WINDOW_HALF_WIDTH, yr + WL_WINDOW_HALF_WIDTH

        if start_yr < year_min or end_yr > year_max:
            _log(
                f"  skipping wl={wl}: window {start_yr}-{end_yr} not fully "
                f"covered by available years {year_min}-{year_max}"
            )
            continue

        da_window = da.sel(time=slice(f"{start_yr}", f"{end_yr}"))
        result = _window_stats(da_window)
        result = result.expand_dims(wl=[wl])
        per_wl_results.append(result)

    if not per_wl_results:
        raise ValueError("No warming level had a fully-covered 21-year window.")

    return xr.concat(per_wl_results, dim="wl")


def compute_wl_diff(da_stats: xr.DataArray) -> xr.DataArray:
    """Difference of each warming level's stats from the first (baseline) level.

    `da_stats` is the (wl, stat, latitude, longitude) result of
    `compute_wl_stats()`. The first entry along `wl` (whatever warming level
    that happens to be -- normally 0.5, but could differ if an earlier level
    was skipped for insufficient window coverage) is treated as the
    baseline: `diff(wl) = value(wl) - value(baseline)`. The baseline's own
    diff slice is set to NaN rather than 0, since "difference from itself"
    isn't a meaningful anomaly value.
    """
    baseline = da_stats.isel(wl=0)
    diff = da_stats - baseline

    wl_position = xr.DataArray(
        np.arange(da_stats.sizes["wl"]), dims="wl", coords={"wl": da_stats.wl}
    )
    return diff.where(wl_position > 0)
