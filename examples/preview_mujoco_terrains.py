"""
Render each terrain as a MuJoCo offscreen screenshot and save a comparison grid.
Usage:  python3 examples/preview_mujoco_terrains.py [--z_scale 0.30]
Saves:  mujoco_terrain_preview.png  in the repo root.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco as mj

REPO       = Path(__file__).resolve().parents[1]
SCENE_TMPL = REPO / "models" / "MJCF" / "go2" / "scene_test_forest.xml"
SCENE_DIR  = SCENE_TMPL.parent
TERRAIN_DIR = REPO / "models" / "MJCF" / "assets" / "terrain"

# (png_path, default_z_scale)
TERRAINS = {
    "forest":        (REPO / "models/MJCF/go2/assets/terrain/forest_hfield.png",    0.12),
    "bc_093p056":    (TERRAIN_DIR / "bc_093p056_xli1m_utm10_20240610_20240621.png", 1.50),
    "bc_093p066":    (TERRAIN_DIR / "bc_093p066_1_1_4_xli1m_utm10_20240610_20240621.png", 1.50),
    "bc_094a059sub": (TERRAIN_DIR / "bc_094a059_2_4_4_xli1m_utm10_20240628_20240628.png", 1.50),
    "bc_094a059":    (TERRAIN_DIR / "bc_094a059_xli1m_utm10_20240611_20240620.png",  1.50),
    "bc_094a060":    (TERRAIN_DIR / "bc_094a060_xli1m_utm10_20240610_20240611.png",  1.50),
    "lidar":         (TERRAIN_DIR / "lidar.png",                                     1.50),
}


def make_scene_xml(terrain_png: Path, z_scale: float) -> Path:
    src = SCENE_TMPL.read_text()
    rel = os.path.relpath(terrain_png, SCENE_DIR / "assets")
    patched = re.sub(
        r'(<hfield\b[^>]*\bfile=")[^"]*(")',
        lambda m: m.group(1) + rel + m.group(2),
        src,
    )
    patched = re.sub(
        r'(<hfield\b[^>]*size="\d+\.?\d* \d+\.?\d*) \d+\.?\d*( \d+\.?\d*")',
        lambda m: f'{m.group(1)} {z_scale}{m.group(2)}',
        patched,
    )
    # Ensure offscreen buffer is large enough
    patched = re.sub(
        r'(<global\b[^>]*?)(\s*/>)',
        lambda m: m.group(1) + ' offwidth="1280" offheight="720"' + m.group(2)
        if 'offwidth' not in m.group(1) else m.group(0),
        patched,
    )
    out = SCENE_DIR / f"_prev_{terrain_png.stem}.xml"
    out.write_text(patched)
    return out


def render_terrain(terrain_png: Path, z_scale: float) -> np.ndarray:
    xml_path = make_scene_xml(terrain_png, z_scale)
    try:
        model = mj.MjModel.from_xml_path(str(xml_path))
        data  = mj.MjData(model)
        # Step forward a tiny bit so hfield geometry is initialised
        mj.mj_forward(model, data)

        renderer = mj.Renderer(model, height=480, width=640)
        cam = mj.MjvCamera()
        mj.mjv_defaultCamera(cam)
        # Angled view — close enough to read surface relief clearly
        cam.type      = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.25]
        cam.distance  = 8.0
        cam.elevation = -30.0
        cam.azimuth   = -120.0

        renderer.update_scene(data, camera=cam)
        frame = renderer.render().copy()
        renderer.close()
        return frame
    finally:
        xml_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z_scale", type=float, default=0.30,
                        help="z_scale for all BC terrains (forest keeps 0.12)")
    args = parser.parse_args()
    bc_z = args.z_scale

    labels, frames = [], []
    for label, (png, default_z) in TERRAINS.items():
        if not png.exists():
            print(f"[SKIP] {label}: {png} not found")
            continue
        z = default_z if default_z is not None else bc_z
        print(f"[RENDER] {label}  z_scale={z:.3f} ...", flush=True)
        frame = render_terrain(png, z)
        labels.append(f"{label}\nz={z:.2f}m")
        frames.append(frame)

    n    = len(frames)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig.suptitle(f"MuJoCo terrain previews  (BC z_scale={bc_z:.2f} m)", fontsize=13)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, frame, lbl in zip(axes, frames, labels):
        ax.imshow(frame)
        ax.set_title(lbl, fontsize=9)
        ax.axis("off")
    for ax in axes[len(frames):]:
        ax.axis("off")

    plt.tight_layout()
    out = REPO / "mujoco_terrain_preview.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
