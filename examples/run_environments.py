#!/usr/bin/env python3
"""
Run case0_full_system on each of the 4 generated environment XMLs.

Produces the same outputs as the ablation study (metrics.json, mppi_debug.mp4,
.npy files) but for different terrains.

Usage:
    python examples/run_environments.py
    python examples/run_environments.py --no-video
    python examples/run_environments.py --force
    python examples/run_environments.py --env bc_093f077_xli1m_utm10_2019
"""
import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import time
import json
import argparse
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ablation import (
    AblationConfig,
    run_simulation,
    save_results,
)

REPO = Path(__file__).resolve().parent.parent
ENV_DIR = REPO / "models" / "MJCF" / "go2" / "environments"
OUTPUT_DIR = "environment_results"


def find_environment_xmls():
    """Find all generated environment XMLs."""
    xmls = sorted(ENV_DIR.glob("*.xml"))
    if not xmls:
        print(f"No .xml files found in {ENV_DIR}")
        print("Run models/MJCF/scripts/generate_environments.py first.")
        sys.exit(1)
    return xmls


def case_already_done(case_name, output_dir):
    case_dir = os.path.join(output_dir, case_name)
    required = ["metrics.json", "time_log_render.npy",
                "q_log_render.npy", "tau_log_render.npy"]
    return all(os.path.exists(os.path.join(case_dir, f)) for f in required)


def main():
    parser = argparse.ArgumentParser(
        description="Run full system on generated environment terrains")
    parser.add_argument("--env", type=str,
                        help="Run a single environment by stem name")
    parser.add_argument("--list", action="store_true",
                        help="List available environments")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip MPPI video rendering (faster)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist")
    args = parser.parse_args()

    xml_files = find_environment_xmls()

    if args.list:
        print("Available environments:")
        for xml in xml_files:
            print(f"  {xml.stem}")
        return

    if args.env:
        xml_files = [x for x in xml_files if x.stem == args.env]
        if not xml_files:
            print(f"Unknown environment: {args.env}")
            print("Use --list to see available environments.")
            return

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}

    for i, xml_path in enumerate(xml_files):
        case_name = xml_path.stem

        print(f"\n{'=' * 70}")
        print(f"  ENVIRONMENT {i + 1}/{len(xml_files)}: {case_name}")
        print(f"  Scene XML: {xml_path}")
        print(f"{'=' * 70}")

        if not args.force and case_already_done(case_name, args.output_dir):
            print(f"  SKIP (results already exist). Use --force to re-run.")
            metrics_path = os.path.join(args.output_dir, case_name, "metrics.json")
            with open(metrics_path) as f:
                summary[case_name] = json.load(f)
            continue

        config = AblationConfig(
            case_name=case_name,
            scene_xml=str(xml_path),
            sim_length_s=20.0,
        )

        t0 = time.perf_counter()
        results = run_simulation(config)
        elapsed = time.perf_counter() - t0

        m = results["metrics"]
        print(f"\n  Completed in {elapsed:.1f}s")
        print(f"  Goal reached: {m['goal_reached']}")
        print(f"  Traversal time: {m['traversal_time_s']}")
        print(f"  Roll RMS: {m['roll_rms_rad']:.4f} rad "
              f"({m['mean_abs_roll_deg']:.2f} deg mean)")
        print(f"  Body contacts: {m['num_body_contacts']} "
              f"({m['body_contact_steps']} steps)")
        print(f"  Dist to goal: {m['dist_to_goal_final_m']:.3f} m")

        save_results(config, results, args.output_dir,
                     save_video=(not args.no_video))
        summary[case_name] = m

        del results
        gc.collect()

    # Save combined summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  ALL ENVIRONMENTS COMPLETE")
    print(f"  Results saved to: {args.output_dir}/")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 70}")

    # Print summary table
    print(f"\n{'Environment':<45} {'Goal?':>6} {'Time(s)':>8} "
          f"{'Roll(deg)':>10} {'Contacts':>9} {'Dist(m)':>8}")
    print("-" * 88)
    for name, m in summary.items():
        t = f"{m['traversal_time_s']:.1f}" if m["traversal_time_s"] else "N/A"
        print(f"{name:<45} {str(m['goal_reached']):>6} {t:>8} "
              f"{m['mean_abs_roll_deg']:>10.2f} "
              f"{m['num_body_contacts']:>9} "
              f"{m['dist_to_goal_final_m']:>8.3f}")


if __name__ == "__main__":
    main()
