MODEL = "MPI-ESM1-2-HR"
SCENARIO = "ssp585"

GCS_INPUT_TEMPLATE = (
    "gs://clim_data_reg_useast1/cmip6_downscaled_woodwell"
    "/{dir}/{prefix}_{model}_ww-isimip_{scenario}_{freq}.zarr"
)
GCS_OUTPUT_TEMPLATE = (
    "gs://clim_data_reg_useast1/cmip6_downscaled_woodwell/annual_aggregates"
    "/{metric}/{metric}_{model}_ww-isimip_{scenario}_day.zarr"
)

# Local CSV mapping warming levels (wl) to the calendar year (yr) at which each
# level is reached, for a given model/scenario. Columns: wl, yr.
WLS_YRS_CSV_TEMPLATE = "wls_yrs_{model}_{scenario}.csv"

# Output path for per-metric warming-level statistics (wl x stat x lat x lon).
GCS_WL_STATS_TEMPLATE = (
    "gs://clim_data_reg_useast1/cmip6_downscaled_woodwell/warming_levels_aggregates"
    "/{metric}/{metric}_{model}_ww-isimip_{scenario}_wls.zarr"
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

# Chunk sizes for opening the monthly wb_percentile store (matches its native
# chunking: 60 months = 5 years per chunk, same spatial tile size as CHUNKS).
MONTHLY_CHUNKS = {"time": 60, "latitude": 120, "longitude": 120}

# Per-variable pieces used to fill in GCS_INPUT_TEMPLATE (dir/prefix/freq),
# plus the chunking to use when opening each with open_variable(). The daily
# variables all share the pattern dir="daily/{var}", prefix="{var}",
# freq="day"; wb_percentile is monthly, lives under a differently-named
# directory, and its filename prefix carries its baseline descriptor
# (bl-1977-2007-w12) rather than matching the variable name.
VARIABLE_SOURCES = {
    "tasmax": {
        "dir": "daily/tasmax",
        "prefix": "tasmax",
        "freq": "day",
        "chunks": CHUNKS,
    },
    "tasmin": {
        "dir": "daily/tasmin",
        "prefix": "tasmin",
        "freq": "day",
        "chunks": CHUNKS,
    },
    "tas": {"dir": "daily/tas", "prefix": "tas", "freq": "day", "chunks": CHUNKS},
    "wetbulbmax": {
        "dir": "daily/wetbulbmax",
        "prefix": "wetbulbmax",
        "freq": "day",
        "chunks": CHUNKS,
    },
    "pr": {"dir": "daily/pr", "prefix": "pr", "freq": "day", "chunks": CHUNKS},
    "wb_percentile": {
        "dir": "monthly_aggregates/wb_percentiles",
        "prefix": "wb-percentiles-bl-1977-2007-w12",
        "freq": "mon",
        "chunks": MONTHLY_CHUNKS,
    },
}

# wb_percentile uses a trailing 12-month rolling window (w12) to compute each
# month's percentile, so the first 11 months of the record (Jan-Nov 1961)
# lack the lookback history needed and are NaN. An annual aggregate for 1961
# would be based on at most 1 valid month (Dec) rather than 12, so metrics
# derived from wb_percentile drop 1961 and start reporting at this year.
WB_PERCENTILE_FIRST_VALID_YEAR = 1962

# ─── warming-level statistics ─────────────────────────────────────────────────

# Each warming level's window spans yr - WL_WINDOW_HALF_WIDTH to
# yr + WL_WINDOW_HALF_WIDTH inclusive (21 years total).
WL_WINDOW_HALF_WIDTH = 10

# Quantile fractions computed per warming-level window, using the nearest-rank
# method (rank = ceil(p * n), clamped to [1, n]) rather than interpolation --
# with only 21 observations per window, interpolating between ranks would
# imply more precision than the sample supports. "min"/"max" are the p=0.0/1.0
# nearest-rank quantiles (i.e. the sample min/max) and are named accordingly
# rather than as "p0"/"p100". The 1st/99th percentiles are omitted: with n=21,
# they resolve to nearly the same rank as min/max (rank 1 and 21 respectively)
# and add little information.
QUANTILES = {"min": 0.0, "p5": 0.05, "p50": 0.5, "p95": 0.95, "max": 1.0}

# Order of labels along the output 'stat' dimension: quantiles plus the mean.
STAT_LABELS = list(QUANTILES) + ["mean"]

# Chunk sizes for writing warming-level stats zarr stores. 'wl' and 'stat' are
# small enough to keep as single chunks; spatial chunking matches OUT_CHUNKS.
WL_OUT_CHUNKS = {
    "wl": 1,
    "stat": len(STAT_LABELS),
    "latitude": OUT_CHUNKS["latitude"],
    "longitude": OUT_CHUNKS["longitude"],
}
