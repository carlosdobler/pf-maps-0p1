"""CLI entry point for computing annual climate stress metrics.

Usage
-----
    python run.py days-above-32c
    python run.py days-above-32c days-above-35c
    python run.py all
    python run.py --list
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

from dask.distributed import Client, LocalCluster

# Suppress heartbeat errors logged by workers during normal cluster shutdown
# (race condition: workers try to contact the scheduler after it's already closed).
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)

# Suppress dask's large-graph warning — expected for a global 0.1° grid
# (~25k tasks), not an error.
warnings.filterwarnings(
    "ignore",
    message="Sending large graph",
    category=UserWarning,
    module="distributed",
)

from metrics import METRIC_REGISTRY
from store import open_variable, save_metric


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute annual climate stress metrics from downscaled daily variables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use --list to see all available metric names.",
    )
    parser.add_argument(
        "metrics",
        nargs="*",
        metavar="METRIC",
        help="Metric name(s) to compute, or 'all' to run every metric.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all available metric names and exit.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=os.cpu_count(),
        help=f"Number of local Dask workers (default: {os.cpu_count()}, all available cores).",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="4GB",
        help="Memory limit per Dask worker (default: 4GB).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list:
        print("\n".join(sorted(METRIC_REGISTRY)))
        sys.exit(0)

    if not args.metrics:
        print(
            "Error: at least one metric name (or 'all') is required.", file=sys.stderr
        )
        print("Run with --list to see available names.", file=sys.stderr)
        sys.exit(1)

    # Resolve metric list
    if args.metrics == ["all"]:
        requested = list(METRIC_REGISTRY)
    else:
        unknown = [m for m in args.metrics if m not in METRIC_REGISTRY]
        if unknown:
            print(f"Error: unknown metric(s): {', '.join(unknown)}", file=sys.stderr)
            print("Run with --list to see available names.", file=sys.stderr)
            sys.exit(1)
        requested = args.metrics

    # Collect required variables (deduplicated across all requested metrics)
    needed_vars: set[str] = set()
    for name in requested:
        needed_vars.update(METRIC_REGISTRY[name]["variables"])

    # Start local Dask cluster
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=1,
        memory_limit=args.memory_limit,
    )
    client = Client(cluster)
    _log(f"Dask dashboard: {client.dashboard_link}")

    try:
        _log(f"Opening variable(s): {', '.join(sorted(needed_vars))}")
        loaded = {var: open_variable(var) for var in needed_vars}

        for i, name in enumerate(requested, 1):
            _log(f"[{i}/{len(requested)}] Computing: {name}")
            t0 = datetime.now()

            variables = METRIC_REGISTRY[name]["variables"]
            fn = METRIC_REGISTRY[name]["fn"]

            result = fn(loaded)

            # Derive the land/ocean mask from the first time step of the first
            # input variable. Computed eagerly — just one zarr chunk read.
            complete = loaded[variables[0]].isel(time=0).notnull().compute()
            result = result.where(complete)

            save_metric(result, name)
            elapsed = (datetime.now() - t0).total_seconds() / 60
            _log(f"[{i}/{len(requested)}] Done: {name} ({elapsed:.2f} min)")

    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
