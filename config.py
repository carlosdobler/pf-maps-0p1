MODEL = "MPI-ESM1-2-HR"
SCENARIO = "ssp585"

GCS_INPUT_TEMPLATE = (
    "gs://clim_data_reg_useast1/cmip6_downscaled_woodwell/daily"
    "/{var}/{var}_{model}_ww-isimip_{scenario}_day.zarr"
)
GCS_OUTPUT_TEMPLATE = (
    "gs://clim_data_reg_useast1/cmip6_downscaled_woodwell/annual_aggregates"
    "/{metric}/{metric}_{model}_ww-isimip_{scenario}_day.zarr"
)

# Chunk sizes for opening input zarr stores (daily data).
#
# ten-hottest-days/nights/wbmax-days and wettest-90-days all process one
# calendar year at a time rather than rechunking the full time series to a
# single chunk (see _top_n_mean and wettest_90_days in metrics.py). Each
# task only needs roughly n_days_per_year * 120 * 120 * 4 bytes ≈ 21 MB
# (up to ~454 days for wettest-90-days, to allow for its trailing lookback
# buffer and non-overlap state across year boundaries, ≈ 26 MB), at the cost
# of a larger dask graph (one task per spatial tile per year). --n-workers
# no longer needs to be throttled for these metrics on that account. Since
# per-task memory is now small, there's headroom to enlarge the spatial tile
# size below to trade fewer/bigger tasks for reduced graph overhead — worth
# revisiting once the current settings are validated.
CHUNKS = {"time": 1000, "latitude": 120, "longitude": 120}

# Chunk sizes for writing output zarr stores (annual data).
OUT_CHUNKS = {"time": 10, "latitude": 120, "longitude": 120}

# Name of the latitude coordinate in the input zarr stores.
LAT_COORD = "latitude"
