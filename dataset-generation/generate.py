#!/usr/bin/env python3
"""
BEAM Artifact: Workload Generator

Generates realistic LLM inference request traces for artifact evaluation.
Default parameters match the primary BEAM use case:
  Category=language, model=m-small, rate=3 RPS, window=19:00-20:00, day=1
"""

import argparse
import contextlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from servegen import Category, ClientPool, generate_workload
from servegen.utils import get_bounded_rate_fn

SERVEGEN_ROOT = Path(__file__).resolve().parent

CATEGORY_MAP = {
    "language": Category.LANGUAGE,
    "reason": Category.REASON,
    "multimodal": Category.MULTIMODAL,
}

CAT_SUFFIX = {
    "language": "lang",
    "reason": "reason",
    "multimodal": "mm",
}


@contextlib.contextmanager
def _repo_root_cwd():
    original = os.getcwd()
    os.chdir(SERVEGEN_ROOT)
    try:
        yield
    finally:
        os.chdir(original)


def parse_hhmm(s):
    try:
        h, m = s.split(":")
        return int(h) * 3600 + int(m) * 60
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected HH:MM, got: {s!r}")


def build_filename(cat_suffix, model, day, start_s, end_s, duration, rate):
    start_h, start_m = start_s // 3600, (start_s % 3600) // 60
    end_h, end_m = end_s // 3600, (end_s % 3600) // 60
    start_str = f"{start_h:02d}h{start_m:02d}m"
    end_str = f"{end_h:02d}h{end_m:02d}m"
    rate_str = str(int(rate)) if rate == int(rate) else str(rate)
    return f"requests_{cat_suffix}_{model}_day{day}_{start_str}-{end_str}_{duration}s_{rate_str}rps.csv"


def save_csv(requests, path):
    data = []
    for r in requests:
        dt = datetime.fromtimestamp(r.timestamp, tz=timezone.utc)
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f%z")
        data.append({
            "TIMESTAMP": ts_str,
            "ContextTokens": r.data["input_tokens"],
            "GeneratedTokens": r.data["output_tokens"],
            "timestamp": ts_str,
        })
    pd.DataFrame(data).to_csv(path, index=False)


def generate(args):
    category_enum = CATEGORY_MAP[args.category]
    cat_suffix = CAT_SUFFIX[args.category]

    start_s = args.start
    end_s = args.end
    duration = args.duration if args.duration is not None else (end_s - start_s)

    # Pre-flight: validate data exists
    data_path = SERVEGEN_ROOT / "data" / args.category / args.model
    if not data_path.exists():
        cat_data_dir = SERVEGEN_ROOT / "data" / args.category
        if cat_data_dir.exists():
            available = [d.name for d in cat_data_dir.iterdir() if d.is_dir()]
        else:
            available = []
        sys.exit(
            f"No data for category='{args.category}', model='{args.model}'.\n"
            f"Available: {available}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = build_filename(cat_suffix, args.model, args.day, start_s, end_s, duration, args.rate)
    csv_path = output_dir / filename

    if csv_path.exists() and not args.force:
        print(f"Found existing dataset: {csv_path}")
        print("Use --force to regenerate.")
        return

    print(f"Generating workload: category={args.category}, model={args.model}, "
          f"rate={args.rate} RPS, day={args.day}, "
          f"window={args.start//3600:02d}:{(args.start%3600)//60:02d}"
          f"-{args.end//3600:02d}:{(args.end%3600)//60:02d}")

    with _repo_root_cwd():
        pool = ClientPool(category_enum, args.model)

    print(f"Loaded {len(pool.clients)} clients")

    day_start = args.day * 86400
    day_end = day_start + 86400
    full_day_view = pool.span(day_start, day_end)
    print(f"Full day view: {len(full_day_view.get())} windows")

    rate_fn = get_bounded_rate_fn(full_day_view, args.rate)
    full_requests = generate_workload(full_day_view, rate_fn, duration=86400, seed=args.seed)
    print(f"Generated {len(full_requests)} requests for full day")

    requests = [r for r in full_requests if start_s <= r.timestamp < end_s]
    print(f"Filtered to {len(requests)} requests in window")

    if not requests:
        print("Warning: no requests in the specified window. Check --start/--end and --day.")

    save_csv(requests, csv_path)
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="BEAM Artifact Workload Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--category", choices=["language", "reason", "multimodal"],
                        default="language", help="Request category")
    parser.add_argument("--model", default="m-small", help="Model name")
    parser.add_argument("--rate", type=float, default=3.0, help="Requests per second")
    parser.add_argument("--start", type=parse_hhmm, default="19:00",
                        metavar="HH:MM", help="Window start time")
    parser.add_argument("--end", type=parse_hhmm, default="19:10",
                        metavar="HH:MM", help="Window end time")
    parser.add_argument("--day", type=int, default=1, help="Day index (0-based)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Override duration (seconds); default: end-start")
    parser.add_argument("--output-dir", default="./datasets", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--force", action="store_true", help="Regenerate even if CSV exists")

    args = parser.parse_args()

    if args.start >= args.end and args.duration is None:
        parser.error("--start must be before --end (or provide --duration)")

    generate(args)


if __name__ == "__main__":
    main()
