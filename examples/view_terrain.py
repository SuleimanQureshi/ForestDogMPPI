"""
Open any terrain in the interactive MuJoCo viewer.

Usage:
    python3 examples/view_terrain.py                        # forest (default)
    python3 examples/view_terrain.py bc_093p066             # specific BC terrain
    python3 examples/view_terrain.py bc_094a059sub          # sub-tile terrain

Available terrain labels:
    forest  bc_093p056  bc_093p066  bc_094a059sub  bc_094a059  bc_094a060  lidar

Controls inside MuJoCo viewer:
    Left-drag   : rotate
    Right-drag  : pan
    Scroll      : zoom
    Ctrl+A      : show/hide axes
    F           : fullscreen
"""
import os
import re
import sys
from pathlib import Path
import mujoco
import mujoco.viewer

REPO        = Path(__file__).resolve().parents[1]
SCENE_DIR   = REPO / "models" / "MJCF" / "go2"
SCENE_TMPL  = SCENE_DIR / "scene_test_forest.xml"
TERRAIN_DIR = REPO / "models" / "MJCF" / "assets" / "terrain"

TERRAINS = {
    "forest":        (REPO / "models/MJCF/go2/assets/terrain/forest_hfield.png",    None),
    "bc_093p056":    (TERRAIN_DIR / "bc_093p056_xli1m_utm10_20240610_20240621.png", 1.50),
    "bc_093p066":    (TERRAIN_DIR / "bc_093p066_1_1_4_xli1m_utm10_20240610_20240621.png", 1.50),
    "bc_094a059sub": (TERRAIN_DIR / "bc_094a059_2_4_4_xli1m_utm10_20240628_20240628.png", 1.50),
    "bc_094a059":    (TERRAIN_DIR / "bc_094a059_xli1m_utm10_20240611_20240620.png",  1.50),
    "bc_094a060":    (TERRAIN_DIR / "bc_094a060_xli1m_utm10_20240610_20240611.png",  1.50),
    "lidar":         (TERRAIN_DIR / "lidar.png",                                     1.50),
}

label = sys.argv[1] if len(sys.argv) > 1 else "forest"

if label not in TERRAINS:
    print(f"Unknown terrain '{label}'. Choose from: {', '.join(TERRAINS)}")
    sys.exit(1)

png_path, z_scale = TERRAINS[label]
if not png_path.exists():
    print(f"PNG not found: {png_path}")
    sys.exit(1)

# Patch the scene template
src = SCENE_TMPL.read_text()
rel = os.path.relpath(png_path, SCENE_DIR / "assets")
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

tmp_xml = SCENE_DIR / f"_view_{label}.xml"
tmp_xml.write_text(patched)

print(f"Loading terrain: {label}  (z_scale={z_scale or 'template default'})")
print("Close the viewer window to exit.")

try:
    model = mujoco.MjModel.from_xml_path(str(tmp_xml))
    data  = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)
finally:
    tmp_xml.unlink(missing_ok=True)
