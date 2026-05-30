"""Re-render the 4 failed cases with their correct terrain XMLs."""
import os
os.environ["MPLBACKEND"] = "Agg"

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_terrain_experiment import make_scene_xml, TERRAINS
from render_3d_videos import render_case

CASES = [
    ("bc_094a059sub", "environment_results_terrain_seed3"),
    ("bc_094a059",    "environment_results_terrain_seed2"),
    ("bc_094a060",    "environment_results_terrain_seed3"),
    ("lidar",         "environment_results_terrain_seed2"),
]

for case_name, output_dir in CASES:
    png_path, z_scale = TERRAINS[case_name]
    case_dir = REPO / output_dir / case_name
    out_mp4  = str(case_dir / "simulation_3d.mp4")

    print(f"\n[{case_name} / {output_dir}]")
    print(f"  terrain: {png_path.name}  z_scale={z_scale}")

    scene_xml = make_scene_xml(png_path, z_scale)
    try:
        render_case(str(case_dir), out_mp4, xml_path=str(scene_xml))
    finally:
        scene_xml.unlink(missing_ok=True)

print("\nAll done.")
