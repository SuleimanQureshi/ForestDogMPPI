#!/usr/bin/env python3
"""
Generate MuJoCo environment XMLs from GeoTIFF elevation files.

For each .tif in models/MJCF/go2/environments/:
  1. Converts it to a PNG heightfield (reusing inspect_tif.py logic)
  2. Generates a scene XML with randomized tree/rock/log positions

Usage:
    python models/MJCF/scripts/generate_environments.py
    python models/MJCF/scripts/generate_environments.py --seed 42
    python models/MJCF/scripts/generate_environments.py --no-png
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


# ─── Heightfield processing (from inspect_tif.py) ────────────────────────────

def process_heightmap(elevation_data, target_size=256, smooth=True):
    """Process raw LIDAR data for MuJoCo heightfield.

    Uses PIL instead of scipy for downsampling/smoothing to avoid version conflicts.
    """
    data = elevation_data.copy().astype(float)
    data[data < -1000] = np.nanmedian(data[data > -1000])
    data = np.nan_to_num(data, nan=np.nanmedian(data))

    # Downsample using PIL (bilinear) instead of scipy.ndimage.zoom
    if data.shape[0] > target_size or data.shape[1] > target_size:
        scale = target_size / max(data.shape)
        new_h = int(data.shape[0] * scale)
        new_w = int(data.shape[1] * scale)
        img = Image.fromarray(data.astype(np.float32), mode='F')
        img = img.resize((new_w, new_h), Image.BILINEAR)
        data = np.array(img)

    # Smooth using a simple box filter (avoids scipy dependency)
    if smooth:
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        pad = kernel_size // 2
        padded = np.pad(data, pad, mode='edge')
        smoothed = np.zeros_like(data)
        for i in range(kernel_size):
            for j in range(kernel_size):
                smoothed += kernel[i, j] * padded[i:i+data.shape[0], j:j+data.shape[1]]
        data = smoothed

    data_min, data_max = np.min(data), np.max(data)
    height_range = 10.0
    normalized = (data - data_min) / (data_max - data_min) * height_range
    return normalized.astype(np.float32)


def tif_to_png(tif_path: Path, png_path: Path):
    """Convert a GeoTIFF to a grayscale PNG heightfield."""
    with rasterio.open(tif_path) as src:
        elevation_data = src.read(1)
    heightfield = process_heightmap(elevation_data, target_size=256)
    heightfield_png = ((heightfield / heightfield.max()) * 255).astype(np.uint8)
    Image.fromarray(heightfield_png).save(png_path)
    print(f"  PNG saved: {png_path.name}  (shape {heightfield_png.shape})")


# ─── Random obstacle generation ──────────────────────────────────────────────

def _reject_near_endpoints(rng, x_lo, x_hi, y_lo, y_hi, min_dist=0.8):
    """Sample (x, y) in range, rejecting points too close to start (-2,0) or goal (3,0)."""
    for _ in range(50):
        x = rng.uniform(x_lo, x_hi)
        y = rng.uniform(y_lo, y_hi)
        d_start = np.sqrt((x + 2) ** 2 + y ** 2)
        d_goal = np.sqrt((x - 3) ** 2 + y ** 2)
        if d_start > min_dist and d_goal > min_dist:
            return x, y
    return x, y  # fallback after retries


def generate_obstacles(rng: np.random.Generator):
    """Generate randomized trees, rocks, and logs that block the direct path.

    Robot goes from (-2, 0) to (3, 0). Obstacles are placed in the corridor
    to force detours, but kept away from the start and goal positions.
      - 8 trees in the mid-corridor (x: -0.5 to 2.5, |y| < 1.2)
      - 4 trees flanking the corridor to narrow passages
      - 3 background trees for realism
    """
    trees = []

    # --- Blocking trees: in the middle section of the path ---
    for i in range(8):
        x, y = _reject_near_endpoints(rng, -0.5, 2.5, -1.2, 1.2)
        radius = rng.uniform(0.10, 0.17)
        half_height = rng.uniform(0.9, 1.6)
        trees.append((f"trunk{i+1}", x, y, radius, half_height))

    # --- Flanking trees: narrow the corridor from the sides ---
    for i in range(4):
        x, _ = _reject_near_endpoints(rng, -0.5, 2.5, -0.5, 0.5)
        side = rng.choice([-1, 1])
        y = side * rng.uniform(1.0, 2.5)
        radius = rng.uniform(0.12, 0.17)
        half_height = rng.uniform(1.0, 1.6)
        trees.append((f"trunk{i+9}", x, y, radius, half_height))

    # --- Background trees for realism ---
    for i in range(3):
        x = rng.uniform(-6, 6)
        y = rng.choice([-1, 1]) * rng.uniform(3.0, 7.0)
        radius = rng.uniform(0.13, 0.17)
        half_height = rng.uniform(1.0, 1.6)
        trees.append((f"trunk{i+13}", x, y, radius, half_height))

    # --- Rocks: in the mid-corridor ---
    rocks = []
    for i in range(3):
        x, y = _reject_near_endpoints(rng, -0.5, 2.5, -1.0, 1.0)
        size = rng.uniform(0.14, 0.22)
        rocks.append((f"rock{i+1}", x, y, size))

    # --- Fallen logs: across the mid-path ---
    logs = []
    for i in range(2):
        x, y = _reject_near_endpoints(rng, 0.0, 2.0, -1.5, 1.5)
        radius = rng.uniform(0.08, 0.12)
        half_length = rng.uniform(0.8, 1.2)
        yaw = rng.uniform(-1.5, 1.5)
        logs.append((f"log{i+1}", x, y, radius, half_length, yaw))

    return trees, rocks, logs


# ─── XML generation ──────────────────────────────────────────────────────────

def generate_xml(png_filename: str, trees, rocks, logs) -> str:
    """Generate a MuJoCo scene XML string matching scene_test_forest_updated.xml structure."""
    tree_lines = []
    for name, x, y, radius, half_height in trees:
        tree_lines.append(
            f'    <geom class="trunk" name="{name}" pos="{x:.2f} {y:.2f} 0" '
            f'size="{radius:.2f} {half_height:.1f}"/>'
        )

    rock_lines = []
    for name, x, y, size in rocks:
        rock_lines.append(
            f'    <geom class="rock" name="{name}" pos="{x:.2f} {y:.2f} 0" '
            f'size="{size:.2f}"/>'
        )

    log_lines = []
    for name, x, y, radius, half_length, yaw in logs:
        log_lines.append(
            f'    <geom class="log" name="{name}"\n'
            f'          pos="{x:.2f} {y:.2f} 0.10"\n'
            f'          size="{radius:.2f} {half_length:.1f}"\n'
            f'          euler="0 0 {yaw:.2f}"/>'
        )

    trees_block = "\n".join(tree_lines)
    rocks_block = "\n".join(rock_lines)
    logs_block = "\n".join(log_lines)

    return f"""\
  <mujoco model="go2 scene">
    <include file="../go2.xml"/>

    <statistic center="0 0 0.1" extent="0.8"/>

    <visual>
      <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
      <rgba haze="0.15 0.25 0.35 1"/>
      <global azimuth="-130" elevation="-20"/>
    </visual>

    <asset>
      <material name="green_ground" rgba="0.15 0.45 0.20 1" reflectance="0.1"/>
      <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
      <texture type="2d" name="groundplane" builtin="checker" mark="edge"
        rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
      <hfield name="forest" size="10.0 10.0 0.75 0.1" file="../environments/{png_filename}"/>
    </asset>

    <worldbody>
      <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.5 0.5 0.5" specular="0.3 0.3 0.3"
        directional="true" castshadow="false"/>
      <geom name="ground" type="hfield" hfield="forest" pos="0 0 -0.2"
        material="green_ground" friction="1.0 0.005 0.0001"/>

      <!-- Trees -->
{trees_block}

      <!-- Rocks -->
{rocks_block}

      <!-- Fallen logs -->
{logs_block}
    </worldbody>

    <default>
      <default class="prop">
        <geom type="box" rgba="0 0.4 1 1"/>
      </default>
      <default class="trunk">
        <geom type="cylinder" rgba="0.35 0.22 0.12 1" friction="1 0.005 0.0001"
          contype="1" conaffinity="1"/>
      </default>
      <default class="rock">
        <geom type="sphere" rgba="0.4 0.4 0.4 1" friction="1 0.01 0.0001"/>
      </default>
      <default class="log">
        <geom type="capsule" rgba="0.30 0.20 0.12 1" friction="1 0.01 0.0001"
          contype="1" conaffinity="1"/>
      </default>
    </default>
  </mujoco>
"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MuJoCo environments from GeoTIFF elevation files")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed (default: hash of each filename)")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip PNG conversion (use existing PNGs)")
    args = parser.parse_args()

    env_dir = Path(__file__).resolve().parent.parent / "go2" / "environments"
    tif_files = sorted(env_dir.glob("*.tif"))

    if not tif_files:
        print(f"No .tif files found in {env_dir}")
        return

    print(f"Found {len(tif_files)} TIF files in {env_dir}\n")

    for idx, tif_path in enumerate(tif_files):
        stem = tif_path.stem
        png_path = env_dir / f"{stem}.png"
        xml_path = env_dir / f"{stem}.xml"

        print(f"Processing: {tif_path.name}")

        # Step 1: TIF → PNG
        if args.no_png and png_path.exists():
            print(f"  PNG exists, skipping: {png_path.name}")
        else:
            tif_to_png(tif_path, png_path)

        # Step 2: Generate XML with randomized obstacles
        if args.seed is not None:
            seed = args.seed
        else:
            # Combine filename hash with a time-based component for unique seeds each run
            base = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16)
            import time as _time
            seed = (base + int(_time.time() * 1000) + idx) % (2**32)

        rng = np.random.default_rng(seed)
        trees, rocks, logs = generate_obstacles(rng)
        xml_content = generate_xml(f"{stem}.png", trees, rocks, logs)
        xml_path.write_text(xml_content)
        print(f"  XML saved: {xml_path.name}  (seed={seed})")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
