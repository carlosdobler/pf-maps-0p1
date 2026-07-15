"""GCS I/O helpers for reading input variables and writing annual metrics."""

from __future__ import annotations

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

    Triggers dask computation. The variable is named using the metric name
    with hyphens replaced by underscores.
    """
    path = GCS_OUTPUT_TEMPLATE.format(metric=metric, model=MODEL, scenario=SCENARIO)
    fs = gcsfs.GCSFileSystem()
    var_name = metric.replace("-", "_")
    da.chunk(OUT_CHUNKS).to_dataset(name=var_name).to_zarr(
        fs.get_mapper(path), mode="w", zarr_format=3, consolidated=False
    )
