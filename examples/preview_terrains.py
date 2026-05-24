"""
Quick visual preview of all terrain heightfield PNGs as 3D surface plots.
Run: python3 examples/preview_terrains.py
Saves terrain_preview.png in the repo root (no display required).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no Qt/Wayland needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from pathlib import Path

REPO        = Path(__file__).resolve().parents[1]
# (png_path, z_scale) — forest keeps its original 0.12, BC terrains use 0.08
TERRAINS = {
    "forest":        (REPO / "models/MJCF/go2/assets/terrain/forest_hfield.png",    0.12),
    "bc_093p056":    (REPO / "models/MJCF/assets/terrain/bc_093p056_xli1m_utm10_20240610_20240621.png", 1.50),
    "bc_093p066":    (REPO / "models/MJCF/assets/terrain/bc_093p066_1_1_4_xli1m_utm10_20240610_20240621.png", 1.50),
    "bc_094a059sub": (REPO / "models/MJCF/assets/terrain/bc_094a059_2_4_4_xli1m_utm10_20240628_20240628.png", 1.50),
    "bc_094a059":    (REPO / "models/MJCF/assets/terrain/bc_094a059_xli1m_utm10_20240611_20240620.png",  1.50),
    "bc_094a060":    (REPO / "models/MJCF/assets/terrain/bc_094a060_xli1m_utm10_20240610_20240611.png",  1.50),
    "lidar":         (REPO / "models/MJCF/assets/terrain/lidar.png",                                     1.50),
}

n = len(TERRAINS)
cols = 4
rows = (n + cols - 1) // cols
fig = plt.figure(figsize=(cols * 5, rows * 4))
fig.suptitle("Terrain heightfields (forest=0.12m, BC LiDAR=0.08m, 12×12m sim)", fontsize=13)

DOWNSAMPLE = 4   # plot every 4th pixel to keep it fast

for i, (label, (png_path, z_scale)) in enumerate(TERRAINS.items()):
    if not png_path.exists():
        print(f"[SKIP] {label}: {png_path} not found")
        continue

    arr = np.array(Image.open(png_path), dtype=float) / 255.0 * z_scale
    arr_ds = arr[::DOWNSAMPLE, ::DOWNSAMPLE]
    N = arr_ds.shape[0]
    x = np.linspace(-6, 6, N)
    y = np.linspace(-6, 6, N)
    X, Y = np.meshgrid(x, y)

    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    ax.plot_surface(X, Y, arr_ds, cmap='terrain', linewidth=0, antialiased=False)
    ax.set_title(f"{label}  (z={z_scale}m)", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_zlim(0, z_scale)
    ax.view_init(elev=35, azim=-60)

plt.tight_layout()
out = REPO / "terrain_preview.png"
plt.savefig(out, dpi=120)
print(f"Saved -> {out}")
