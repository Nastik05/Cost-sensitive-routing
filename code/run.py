"""One entry point: compute, check reference results, then plot."""
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
from pathlib import Path
from experiment import ROOT, run
from plot import draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    path = args.output / "metrics.csv" if args.plots_only else run(args.output)
    draw(path)


if __name__ == "__main__":
    main()
