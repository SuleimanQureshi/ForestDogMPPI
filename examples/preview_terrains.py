"""
Quick visual preview of all terrain heightfield PNGs as 3D surface plots.
Run: python3 examples/preview_terrains.py
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from pathlib import Path

REPO        = Path(__file__).resolve().parents[1]
Z_SCALE     = 0.40   # metres, matches scene_test_forest.xml

TERRAINS = {
    "forest":        REPO / "models/MJCF/go2/assets/terrain/forest_hfield.png",
    "bc_093p056":    REPO / "models/MJCF/assets/terrain/bc_093p056_xli1m_utm10_20240610_20240621.png",
    "bc_093p066":    REPO / "models/MJCF/assets/terrain/bc_093p066_1_1_4_xli1m_utm10_20240610_20240621.png",
    "bc_094a059sub": REPO / "models/MJCF/assets/terrain/bc_094a059_2_4_4_xli1m_utm10_20240628_20240628.png",
    "bc_094a059":    REPO / "models/MJCF/assets/terrain/bc_094a059_xli1m_utm10_20240611_20240620.png",
    "bc_094a060":    REPO / "models/MJCF/assets/terrain/bc_094a060_xli1m_utm10_20240610_20240611.png",
    "lidar":         REPO / "models/MJCF/assets/terrain/lidar.png",
}

n = len(TERRAINS)
cols = 4
rows = (n + cols - 1) // cols
fig = plt.figure(figsize=(cols * 5, rows * 4))
fig.suptitle("Terrain heightfields (z_scale=0.40 m, 12×12 m sim window)", fontsize=13)

DOWNSAMPLE = 4   # plot every 4th pixel to keep it fast

for i, (label, png_path) in enumerate(TERRAINS.items()):
    if not png_path.exists():
        print(f"[SKIP] {label}: {png_path} not found")
        continue

    arr = np.array(Image.open(png_path), dtype=float) / 255.0 * Z_SCALE
    arr_ds = arr[::DOWNSAMPLE, ::DOWNSAMPLE]
    N = arr_ds.shape[0]
    x = np.linspace(-6, 6, N)
    y = np.linspace(-6, 6, N)
    X, Y = np.meshgrid(x, y)

    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    ax.plot_surface(X, Y, arr_ds, cmap='terrain', linewidth=0, antialiased=False)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_zlim(0, Z_SCALE)
    ax.view_init(elev=35, azim=-60)

plt.tight_layout()
plt.show()
