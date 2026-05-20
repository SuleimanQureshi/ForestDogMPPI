"""
Terrain traversability experiment runner.
For each terrain PNG, generates a temporary MuJoCo scene XML and runs ex02_trot_forward.py.
Metrics (replan count, path length, goal distance) are printed at the end.
Videos are saved as mppi_debug_<terrain_label>.mp4 in the repo root.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO        = Path(__file__).resolve().parents[1]
SCENE_TMPL  = REPO / "models" / "MJCF" / "go2" / "scene_test_forest.xml"
SCENE_DIR   = SCENE_TMPL.parent          # temp XMLs must live here so go2.xml include resolves
TERRAIN_DIR = REPO / "models" / "MJCF" / "assets" / "terrain"
EX02        = Path(__file__).parent / "ex02_trot_forward.py"
PYTHON      = sys.executable

TERRAINS = {
    "forest":        REPO / "models" / "MJCF" / "go2" / "assets" / "terrain" / "forest_hfield.png",
    "bc_093p056":    TERRAIN_DIR / "bc_093p056_xli1m_utm10_20240610_20240621.png",
    "bc_093p066":    TERRAIN_DIR / "bc_093p066_1_1_4_xli1m_utm10_20240610_20240621.png",
    "bc_094a059sub": TERRAIN_DIR / "bc_094a059_2_4_4_xli1m_utm10_20240628_20240628.png",
    "bc_094a059":    TERRAIN_DIR / "bc_094a059_xli1m_utm10_20240611_20240620.png",
    "bc_094a060":    TERRAIN_DIR / "bc_094a060_xli1m_utm10_20240610_20240611.png",
    "lidar":         TERRAIN_DIR / "lidar.png",
}


def make_scene_xml(terrain_png: Path) -> Path:
    """Write a patched scene XML into models/MJCF/go2/ so include/asset paths resolve.

    go2.xml sets <compiler meshdir="assets"/>, so MuJoCo prepends assets/ to every
    hfield file= path.  We must therefore express the PNG path relative to
    SCENE_DIR/assets/, not SCENE_DIR itself.
    """
    src = SCENE_TMPL.read_text()
    rel = os.path.relpath(terrain_png, SCENE_DIR / "assets")
    # swap PNG path
    patched = re.sub(
        r'(<hfield\b[^>]*\bfile=")[^"]*(")',
        lambda m: m.group(1) + rel + m.group(2),
        src,
    )
    # ensure z_scale=0.40 regardless of what the template says
    patched = re.sub(
        r'(size="\d+\.?\d* \d+\.?\d*) \d+\.?\d* (\d+\.?\d*")',
        lambda m: m.group(1) + ' 0.40 ' + m.group(2),
        patched,
    )
    out = SCENE_DIR / f"_tmp_scene_{terrain_png.stem}.xml"
    out.write_text(patched)
    return out


def run_terrain(label: str, scene_xml: Path) -> dict:
    env = os.environ.copy()
    env["EX02_SCENE_XML"]   = str(scene_xml)
    env["EX02_VIDEO_LABEL"] = label

    metrics = {"label": label, "exit_code": "?"}
    collected = []

    # Stream stdout/stderr live to terminal AND collect lines for metrics parsing
    proc = subprocess.Popen(
        [PYTHON, str(EX02)],
        env=env,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)
        collected.append(line)

    proc.wait()
    metrics["exit_code"] = proc.returncode

    # Parse the metrics line ex02 prints
    for line in collected:
        if line.startswith("EXPERIMENT_METRICS"):
            for token in line.split()[1:]:
                k, v = token.split("=", 1)
                metrics[k] = v
            break

    return metrics


def main():
    results = []

    for label, png_path in TERRAINS.items():
        if not png_path.exists():
            print(f"\n[SKIP] {label}: {png_path} not found\n")
            continue

        print(f"\n{'='*60}")
        print(f"[RUN] terrain={label}  png={png_path.name}")
        print(f"{'='*60}\n")

        scene_xml = make_scene_xml(png_path)
        try:
            metrics = run_terrain(label, scene_xml)
        finally:
            scene_xml.unlink(missing_ok=True)   # always clean up temp XML

        results.append(metrics)
        print(f"\n[DONE] {label}: {metrics}\n")

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    cols = ["label", "replan_count", "path_length_m", "final_dist_to_goal_m", "goal_reached", "foot_obstacle_contacts"]
    print("  ".join(f"{c:<22}" for c in cols))
    print("-" * (22 * len(cols)))
    for r in results:
        print("  ".join(f"{r.get(c, 'N/A'):<22}" for c in cols))


if __name__ == "__main__":
    main()
