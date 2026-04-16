"""
Batch inference script for AAGNet machining feature recognition.

Processes all STEP files in a folder and outputs per-part JSON files
or a single consolidated JSON.

Usage:
    python infer_batch.py path/to/step_folder/
    python infer_batch.py path/to/step_folder/ --output results/
    python infer_batch.py path/to/step_folder/ --consolidated results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from infer import (
    load_model,
    infer_step_file,
    ATTR_SCHEMA_PATH,
    ATTR_STAT_PATH,
)
from utils.data_utils import load_json_or_pkl, load_statistics


def main():
    parser = argparse.ArgumentParser(
        description="AAGNet batch inference: folder of STEP files -> feature segmentation JSONs"
    )
    parser.add_argument(
        "step_folder", type=str, help="Folder containing STEP files (.step/.stp)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output folder for per-part JSON files (default: <step_folder>/aagnet_results/)",
    )
    parser.add_argument(
        "--consolidated",
        "-c",
        type=str,
        default=None,
        help="Path for a single consolidated JSON with all results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--inst-threshold", type=float, default=0.5)
    parser.add_argument("--bottom-threshold", type=float, default=0.5)
    args = parser.parse_args()

    step_folder = Path(args.step_folder)
    if not step_folder.is_dir():
        print(f"Error: {step_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    step_files = sorted(
        set(
            list(step_folder.glob("*.step"))
            + list(step_folder.glob("*.stp"))
            + list(step_folder.glob("*.STEP"))
            + list(step_folder.glob("*.STP"))
        ),
        key=lambda p: p.name.lower(),
    )
    if not step_files:
        print(f"No STEP files found in {step_folder}", file=sys.stderr)
        sys.exit(1)

    # Output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = step_folder / "aagnet_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    print(f"Loading model on {args.device}...")
    attribute_schema = load_json_or_pkl(str(ATTR_SCHEMA_PATH))
    stat = load_statistics(str(ATTR_STAT_PATH))
    model = load_model(args.device)
    print(f"Processing {len(step_files)} STEP files...\n")

    all_results = []
    successes = 0
    failures = 0
    total_time = 0.0

    for i, step_file in enumerate(step_files, 1):
        print(f"[{i}/{len(step_files)}] {step_file.name}...", end=" ", flush=True)
        t0 = time.time()

        result = infer_step_file(
            str(step_file),
            model,
            args.device,
            attribute_schema,
            stat,
            inst_thres=args.inst_threshold,
            bottom_thres=args.bottom_threshold,
        )

        elapsed = time.time() - t0
        total_time += elapsed

        if "error" in result:
            print(f"FAILED ({result['error']})")
            failures += 1
        else:
            n_features = len(result["features"])
            n_faces = result["total_faces"]
            print(f"OK ({n_faces} faces, {n_features} features, {elapsed:.2f}s)")
            successes += 1

        # Write per-part JSON
        out_path = out_dir / f"{step_file.stem}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch complete: {successes} succeeded, {failures} failed")
    print(f"Total time: {total_time:.2f}s ({total_time/len(step_files):.2f}s avg)")
    print(f"Per-part JSONs saved to: {out_dir}")

    # Consolidated output
    if args.consolidated:
        consolidated = {
            "total_files": len(step_files),
            "successes": successes,
            "failures": failures,
            "total_time_s": round(total_time, 2),
            "results": all_results,
        }
        Path(args.consolidated).write_text(
            json.dumps(consolidated, indent=2), encoding="utf-8"
        )
        print(f"Consolidated JSON: {args.consolidated}")


if __name__ == "__main__":
    main()
