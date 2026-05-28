"""
Terrain traversability experiment runner.

For each terrain, builds a temporary MuJoCo scene XML, runs the full simulation
via run_ablation.run_simulation(), saves structured results (metrics.json, .npy
logs, mppi_debug.mp4, simulation_3d.mp4) into environment_results_terrain/, and
writes a top-level summary.json.

Usage:
    python examples/run_terrain_experiment.py                  # full 18-s runs
    python examples/run_terrain_experiment.py --sim-length 1   # 1-s validation
    python examples/run_terrain_experiment.py --no-video       # skip video rendering
    python examples/run_terrain_experiment.py --force          # overwrite existing
"""
import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO        = Path(__file__).resolve().parents[1]
SCENE_TMPL  = REPO / "models" / "MJCF" / "go2" / "scene_test_forest.xml"
SCENE_DIR   = SCENE_TMPL.parent
TERRAIN_DIR = REPO / "models" / "MJCF" / "assets" / "terrain"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_ablation import AblationConfig, run_simulation, save_results
from render_3d_videos import render_case

OUTPUT_DIR = str(REPO / "environment_results_terrain")

# (label, png_path, z_scale)
# forest uses the original synthetic z_scale=0.12; BC LiDAR terrains use 1.50
TERRAINS = {
    "forest":        (REPO / "models/MJCF/go2/assets/terrain/forest_hfield.png",    None),
    "bc_093p056":    (TERRAIN_DIR / "bc_093p056_xli1m_utm10_20240610_20240621.png", 1.50),
    "bc_093p066":    (TERRAIN_DIR / "bc_093p066_1_1_4_xli1m_utm10_20240610_20240621.png", 1.50),
    "bc_094a059sub": (TERRAIN_DIR / "bc_094a059_2_4_4_xli1m_utm10_20240628_20240628.png", 1.50),
    "bc_094a059":    (TERRAIN_DIR / "bc_094a059_xli1m_utm10_20240611_20240620.png",  1.50),
    "bc_094a060":    (TERRAIN_DIR / "bc_094a060_xli1m_utm10_20240610_20240611.png",  1.50),
    "lidar":         (TERRAIN_DIR / "lidar.png",                                     1.50),
}


def make_scene_xml(terrain_png: Path, z_scale: float | None = None) -> Path:
    """Write a patched scene XML into models/MJCF/go2/ so include/asset paths resolve."""
    src = SCENE_TMPL.read_text()
    rel = os.path.relpath(terrain_png, SCENE_DIR / "assets")
    patched = re.sub(
        r'(<hfield\b[^>]*\bfile=")[^"]*(")',
        lambda m: m.group(1) + rel + m.group(2),
        src,
    )
    if z_scale is not None:
        patched = re.sub(
            r'(<hfield\b[^>]*size="\d+\.?\d* \d+\.?\d*) \d+\.?\d*( \d+\.?\d*")',
            lambda m: f'{m.group(1)} {z_scale}{m.group(2)}',
            patched,
        )
    out = SCENE_DIR / f"_tmp_scene_{terrain_png.stem}.xml"
    out.write_text(patched)
    return out


def case_already_done(case_name: str, output_dir: str) -> bool:
    case_dir = Path(output_dir) / case_name
    required = ["metrics.json", "time_log_render.npy", "q_log_render.npy",
                "tau_log_render.npy", "simulation_3d.mp4"]
    return all((case_dir / f).exists() for f in required)


def main():
    parser = argparse.ArgumentParser(
        description="Run terrain traversability experiment and save structured results")
    parser.add_argument("--sim-length", type=float, default=18.0,
                        help="Simulation length in seconds (default: 18.0)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip MPPI debug and 3D video rendering")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}

    terrain_items = [(label, png, zs) for label, (png, zs) in TERRAINS.items()
                     if png.exists()]
    skipped = [label for label, (png, _) in TERRAINS.items() if not png.exists()]
    if skipped:
        print(f"[WARN] Missing PNGs, skipping: {skipped}")

    for label, png_path, z_scale in tqdm(terrain_items, desc="terrains", unit="terrain"):

        if not args.force and case_already_done(label, args.output_dir):
            print(f"\n[SKIP] {label}: results already exist (use --force to re-run)")
            metrics_path = Path(args.output_dir) / label / "metrics.json"
            with open(metrics_path) as f:
                summary[label] = json.load(f)
            continue

        print(f"\n{'=' * 70}")
        print(f"  TERRAIN: {label}  |  png: {png_path.name}  |  z_scale: {z_scale or 'template'}")
        print(f"  sim_length: {args.sim_length}s  ->  {args.output_dir}/{label}/")
        print(f"{'=' * 70}")

        scene_xml = make_scene_xml(png_path, z_scale)
        try:
            config = AblationConfig(
                case_name=label,
                scene_xml=str(scene_xml),
                sim_length_s=args.sim_length,
            )

            t0 = time.perf_counter()
            results = run_simulation(config)
            elapsed = time.perf_counter() - t0

            m = results["metrics"]
            print(f"\n  Completed in {elapsed:.1f}s")
            print(f"  Goal reached: {m['goal_reached']}")
            print(f"  Dist to goal: {m['dist_to_goal_final_m']:.3f} m")
            print(f"  Roll RMS: {m['roll_rms_rad']:.4f} rad  "
                  f"({m['mean_abs_roll_deg']:.2f} deg mean)")
            print(f"  Body contacts: {m['num_body_contacts']} "
                  f"({m['body_contact_steps']} steps)")

            save_results(config, results, args.output_dir,
                         save_video=(not args.no_video))
            summary[label] = m

            # Render simulation_3d.mp4 from saved q_log_render.npy
            if not args.no_video:
                case_dir = str(Path(args.output_dir) / label)
                sim3d_path = str(Path(case_dir) / "simulation_3d.mp4")
                print(f"  Rendering simulation_3d.mp4 ...")
                render_case(case_dir, sim3d_path, xml_path=str(scene_xml))

        finally:
            scene_xml.unlink(missing_ok=True)

        del results
        gc.collect()

    # Top-level summary
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  ALL TERRAINS COMPLETE")
    print(f"  Results: {args.output_dir}/")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 70}\n")

    print(f"{'Terrain':<20} {'Goal?':>6} {'Time(s)':>8} "
          f"{'Roll(deg)':>10} {'Contacts':>9} {'Dist(m)':>8}")
    print("-" * 65)
    for name, m in summary.items():
        t = f"{m['traversal_time_s']:.1f}" if m.get("traversal_time_s") else "N/A"
        print(f"{name:<20} {str(m['goal_reached']):>6} {t:>8} "
              f"{m['mean_abs_roll_deg']:>10.2f} "
              f"{m['num_body_contacts']:>9} "
              f"{m['dist_to_goal_final_m']:>8.3f}")


if __name__ == "__main__":
    main()
