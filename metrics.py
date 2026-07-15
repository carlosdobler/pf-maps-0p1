"""
Annual climate stress metrics.

Each public function accepts a dict of xr.DataArray (keyed by variable name)
and returns an annual xr.DataArray with a 'time' dimension of yearly Jan-1
timestamps.

The METRIC_REGISTRY at the bottom maps each kebab-case metric name to its
function and the list of input variables it requires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from config import LAT_COORD


# ─── private helpers ──────────────────────────────────────────────────────────


def _annual_sum(condition: xr.DataArray) -> xr.DataArray:
    """Count True days per calendar year."""
    return condition.resample(time="YS").sum()


def _top_n_mean(da: xr.DataArray, n: int) -> xr.DataArray:
    """Annual mean of the n highest daily values.

    Uses apply_ufunc with vectorize=True so each (lat, lon) pixel is processed
    independently. Only the time dimension is rechunked to -1 (spatial chunks
    stay at their native tile size) so each dask task receives the full time
    series for its spatial tile without triggering a cross-worker reshuffle.
    See CHUNKS in config.py for the resulting per-task memory tradeoff.
    """
    years = da.time.dt.year.values
    unique_years = np.unique(years)

    def _fn(x: np.ndarray) -> np.ndarray:
        result = np.full(len(unique_years), np.nan)
        for i, yr in enumerate(unique_years):
            vals = x[years == yr]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                k = min(n, len(valid))
                result[i] = np.mean(np.partition(valid, -k)[-k:])
        return result

    result = xr.apply_ufunc(
        _fn,
        da.chunk({"time": -1}),
        input_core_dims=[["time"]],
        output_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"year": len(unique_years)}},
    )
    return result.rename({"year": "time"}).assign_coords(
        time=pd.to_datetime([f"{y}-01-01" for y in unique_years])
    )


# ─── tasmax metrics ───────────────────────────────────────────────────────────


def days_above_32c(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["tasmax"] >= 32.0)


def days_above_35c(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["tasmax"] >= 35.0)


def days_above_38c(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["tasmax"] >= 38.0)


def days_above_45c(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["tasmax"] >= 45.0)


def average_daytime_temperature(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return ds["tasmax"].resample(time="YS").mean()


def ten_hottest_days(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _top_n_mean(ds["tasmax"], 10)


def freezing_days(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["tasmax"] < 0.0)


# ─── tasmin metrics ───────────────────────────────────────────────────────────


def frost_nights(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["tasmin"] < 0.0)


def nights_above_20c(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["tasmin"] >= 20.0)


def nights_above_25c(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["tasmin"] >= 25.0)


def average_nighttime_temperature(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return ds["tasmin"].resample(time="YS").mean()


def ten_hottest_nights(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _top_n_mean(ds["tasmin"], 10)


# ─── tas metrics ─────────────────────────────────────────────────────────────


def average_temperature(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return ds["tas"].resample(time="YS").mean()


def average_winter_temperature(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    """Hemisphere-aware annual winter mean temperature.

    Northern hemisphere (lat >= 0): mean over DJF months of that calendar year
    (i.e. Dec of year Y is averaged with Jan and Feb of year Y, not year Y+1).
    Southern hemisphere (lat < 0): mean over JJA months of that calendar year.
    """
    tas = ds["tas"]
    lat_mask = tas[LAT_COORD] >= 0  # True = northern hemisphere

    north = (
        tas.where(tas.time.dt.month.isin([12, 1, 2]))
        .where(lat_mask)
        .resample(time="YS")
        .mean()
    )
    south = (
        tas.where(tas.time.dt.month.isin([6, 7, 8]))
        .where(~lat_mask)
        .resample(time="YS")
        .mean()
    )
    return xr.where(lat_mask, north, south)


# ─── wet-bulb metrics ─────────────────────────────────────────────────────────


def days_above_26c_wbmax(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["wetbulbmax"] >= 26.0)


def days_above_28c_wbmax(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["wetbulbmax"] >= 28.0)


def days_above_30c_wbmax(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["wetbulbmax"] >= 30.0)


def days_above_32c_wbmax(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _annual_sum(ds["wetbulbmax"] >= 32.0)


def ten_hottest_wbmax_days(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return _top_n_mean(ds["wetbulbmax"], 10)


# ─── precipitation metrics ────────────────────────────────────────────────────


def total_annual_precipitation(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return ds["pr"].resample(time="YS").sum()


def wettest_day(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    return ds["pr"].resample(time="YS").max()


def wettest_90_days(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    """Maximum non-overlapping 90-day precipitation sum per year.

    The 90-day rolling sum is computed over the full time series, so windows
    can span calendar year boundaries (e.g. a window ending in January may
    include days from the previous December). For each calendar year the window
    whose end date falls within that year and whose sum is largest is selected,
    subject to the constraint that it does not overlap with the previous year's
    selected window.
    """
    precip = ds["pr"]
    years = precip.time.dt.year.values
    unique_years = np.unique(years)

    def _fn(x: np.ndarray) -> np.ndarray:
        n = len(x)
        result = np.full(len(unique_years), np.nan)

        if np.all(np.isnan(x)):
            return result

        # 90-day backward rolling sum via cumsum (NaN treated as 0)
        filled = np.where(np.isnan(x), 0.0, x)
        cs = np.concatenate([[0.0], np.cumsum(filled)])
        runsum = np.full(n, np.nan)
        runsum[89:] = cs[90:] - cs[: n - 89]

        prev_max_pos = -90
        for i, yr in enumerate(unique_years):
            positions = np.where(years == yr)[0]
            valid_start = max(int(positions[0]), prev_max_pos + 90)
            valid_end = int(positions[-1])

            if valid_start > valid_end:
                continue

            valid_range = np.arange(valid_start, valid_end + 1)
            max_pos = int(valid_range[np.argmax(runsum[valid_range])])
            result[i] = runsum[max_pos]
            prev_max_pos = max_pos

        return result

    result = xr.apply_ufunc(
        _fn,
        precip.chunk({"time": -1}),
        input_core_dims=[["time"]],
        output_core_dims=[["year"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"year": len(unique_years)}},
    )
    return result.rename({"year": "time"}).assign_coords(
        time=pd.to_datetime([f"{y}-01-01" for y in unique_years])
    )


# ─── multi-variable metrics ───────────────────────────────────────────────────


def snowy_days(ds: dict[str, xr.DataArray]) -> xr.DataArray:
    """Count days with precipitation ≥ 1 mm and mean temperature < 0°C."""
    return _annual_sum((ds["pr"] >= 1.0) & (ds["tas"] < 0.0))


# ─── registry ─────────────────────────────────────────────────────────────────

METRIC_REGISTRY: dict[str, dict] = {
    "days-above-32c": {"fn": days_above_32c, "variables": ["tasmax"]},
    "days-above-35c": {"fn": days_above_35c, "variables": ["tasmax"]},
    "days-above-38c": {"fn": days_above_38c, "variables": ["tasmax"]},
    "days-above-45c": {"fn": days_above_45c, "variables": ["tasmax"]},
    "average-temperature": {"fn": average_temperature, "variables": ["tas"]},
    "average-daytime-temperature": {
        "fn": average_daytime_temperature,
        "variables": ["tasmax"],
    },
    "ten-hottest-days": {"fn": ten_hottest_days, "variables": ["tasmax"]},
    "freezing-days": {"fn": freezing_days, "variables": ["tasmax"]},
    "frost-nights": {"fn": frost_nights, "variables": ["tasmin"]},
    "nights-above-20c": {"fn": nights_above_20c, "variables": ["tasmin"]},
    "nights-above-25c": {"fn": nights_above_25c, "variables": ["tasmin"]},
    "average-nighttime-temperature": {
        "fn": average_nighttime_temperature,
        "variables": ["tasmin"],
    },
    "average-winter-temperature": {
        "fn": average_winter_temperature,
        "variables": ["tas"],
    },
    "ten-hottest-nights": {"fn": ten_hottest_nights, "variables": ["tasmin"]},
    "days-above-26c-wbmax": {"fn": days_above_26c_wbmax, "variables": ["wetbulbmax"]},
    "days-above-28c-wbmax": {"fn": days_above_28c_wbmax, "variables": ["wetbulbmax"]},
    "days-above-30c-wbmax": {"fn": days_above_30c_wbmax, "variables": ["wetbulbmax"]},
    "days-above-32c-wbmax": {"fn": days_above_32c_wbmax, "variables": ["wetbulbmax"]},
    "ten-hottest-wbmax-days": {
        "fn": ten_hottest_wbmax_days,
        "variables": ["wetbulbmax"],
    },
    "total-annual-precipitation": {
        "fn": total_annual_precipitation,
        "variables": ["pr"],
    },
    "wettest-90-days": {"fn": wettest_90_days, "variables": ["pr"]},
    "snowy-days": {"fn": snowy_days, "variables": ["pr", "tas"]},
    "wettest-day": {"fn": wettest_day, "variables": ["pr"]},
}
