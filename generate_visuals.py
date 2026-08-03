"""CLI entry point for generating warming-level map visuals.

For a given metric, reads its warming-level stats zarr store (or the
`diff_<metric>` variable, with `-d`/`--diff`), and renders a 3x3 grid of
maps: warming levels 1.0, 2.0, 3.0 across columns, and the 5th percentile,
median, and 95th percentile across rows. All 9 panels share a single fill
scale, capped to the 5th/95th percentile of the values shown, so panels are
visually comparable. The map resolution is downsampled (coarsened) before
plotting, since the native 0.1 degree grid is far too dense to render
per-pixel. Output is written as a PNG to visuals/.

Usage
-----
    python generate_visuals.py days-above-32c
    python generate_visuals.py days-above-32c -d
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from plotnine import *

import gcsfs

from config import GCS_WL_STATS_TEMPLATE, MODEL, SCENARIO, WL_OUT_CHUNKS
from metrics import METRIC_REGISTRY

# Panels shown: 3 warming levels x 3 stats.
VIS_WARMING_LEVELS = [1.0, 2.0, 3.0]
VIS_STATS = ["p5", "p50", "p95"]
STAT_TITLES = {"p5": "5th percentile", "p50": "median", "p95": "95th percentile"}

# Spatial downsampling factor applied before plotting (mean-aggregated).
COARSEN_FACTOR = 10

# Fraction range used to cap the shared fill scale across all 9 panels.
SCALE_QUANTILES = (0.05, 0.95)

OUTPUT_DIR = "visuals"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 3x3 grid of warming-level maps (wl 1.0/2.0/3.0 x "
            "p5/median/p95) for a metric's warming-level stats."
        ),
    )
    parser.add_argument(
        "metric",
        metavar="METRIC",
        help="Metric name to plot (see metrics.METRIC_REGISTRY for valid names).",
    )
    parser.add_argument(
        "-d",
        "--diff",
        action="store_true",
        help="Plot the diff_<metric> variable (anomaly vs. the wl=0.5 baseline) instead of <metric>.",
    )
    parser.add_argument(
        "--coarsen",
        type=int,
        default=COARSEN_FACTOR,
        help=f"Spatial coarsening factor applied before plotting (default: {COARSEN_FACTOR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to write the PNG to (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args(argv)


def _open_wl_stats_variable(metric: str, var_name: str) -> xr.DataArray:
    """Open a single data variable from a metric's warming-level stats store."""
    path = GCS_WL_STATS_TEMPLATE.format(metric=metric, model=MODEL, scenario=SCENARIO)
    fs = gcsfs.GCSFileSystem()
    return xr.open_zarr(fs.get_mapper(path), chunks=WL_OUT_CHUNKS, consolidated=False)[
        var_name
    ]


def build_plot_data(da: xr.DataArray, coarsen_factor: int) -> pd.DataFrame:
    """Select the 9 (wl, stat) panels, coarsen spatially, and return a tidy DataFrame."""
    da = da.sel(wl=VIS_WARMING_LEVELS, method="nearest").sel(stat=VIS_STATS)

    if coarsen_factor > 1:
        da = da.coarsen(
            latitude=coarsen_factor, longitude=coarsen_factor, boundary="trim"
        ).mean()

    da = da.compute()

    df = da.to_dataframe(name="value").reset_index()

    df["wl_label"] = pd.Categorical(
        [f"WL {v:.1f}\u00b0C" for v in df["wl"]],
        categories=[f"WL {v:.1f}\u00b0C" for v in VIS_WARMING_LEVELS],
        ordered=True,
    )
    df["stat_label"] = pd.Categorical(
        [STAT_TITLES[s] for s in df["stat"]],
        categories=[STAT_TITLES[s] for s in VIS_STATS],
        ordered=True,
    )
    return df


def make_plot(df: pd.DataFrame, metric: str, diff: bool) -> ggplot:
    """Build the faceted map plot with a single shared, clipped fill scale."""
    vmin, vmax = np.nanpercentile(df["value"], [q * 100 for q in SCALE_QUANTILES])

    if diff:
        # Diff values can be positive or negative (a warming level's stat
        # could be lower than the wl=0.5 baseline for some metrics), so
        # center the scale on 0 with a symmetric range and a diverging
        # colormap, rather than viridis's sequential one.
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        cmap_name = "RdBu_r"
    else:
        cmap_name = "viridis"

    # Clip (rather than rely on out-of-bounds handling) so every panel shares
    # exactly the same fill scale, capped at the 5th/95th percentile of all
    # values shown across the 9 panels.
    df = df.copy()
    df["value_clipped"] = df["value"].clip(vmin, vmax)

    title = (
        f"{metric}: difference from wl=0.5 baseline"
        if diff
        else f"{metric}: warming-level statistics"
    )

    return (
        ggplot(df, aes(x="longitude", y="latitude", fill="value_clipped"))
        + geom_tile()
        + facet_grid("stat_label ~ wl_label")
        + scale_fill_cmap(
            cmap_name=cmap_name,
            limits=(vmin, vmax),
            na_value="none",
        )
        + coord_fixed(expand=False)
        + ggtitle(title)
        + theme(
            axis_title=element_blank(),
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            legend_position="bottom",
            legend_key_width=250,
            legend_key_height=10,
            legend_title=element_blank(),
            plot_title=element_text(ha="left"),
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.metric not in METRIC_REGISTRY:
        print(f"Error: unknown metric: {args.metric}", file=sys.stderr)
        print(
            "Available metrics: " + ", ".join(sorted(METRIC_REGISTRY)), file=sys.stderr
        )
        sys.exit(1)

    var_name = args.metric.replace("-", "_")
    if args.diff:
        var_name = f"diff_{var_name}"

    da = _open_wl_stats_variable(args.metric, var_name)
    df = build_plot_data(da, args.coarsen)
    p = make_plot(df, args.metric, args.diff)

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "-diff" if args.diff else ""
    out_path = os.path.join(args.output_dir, f"{args.metric}{suffix}.png")
    p.save(out_path, width=12, height=8, dpi=150, verbose=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
