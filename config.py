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
# For metrics that need the full time series per pixel (ten-hottest-*,
# wettest-90-days), only the time dimension is rechunked to -1 — spatial
# chunks are kept at their native 120x120 tile size.
#
# The tradeoff: each task for those metrics needs
# n_timesteps * 120 * 120 * 4 bytes ≈ 2.9 GB. Limit concurrency accordingly
# via `--n-workers` when running ten-hottest-*/wettest-90-days
# (e.g. floor(available_RAM / 2.9 GB)).
CHUNKS = {"time": 1000, "latitude": 120, "longitude": 120}

# Chunk sizes for writing output zarr stores (annual data).
OUT_CHUNKS = {"time": 10, "latitude": 120, "longitude": 120}

# Name of the latitude coordinate in the input zarr stores.
LAT_COORD = "latitude"
