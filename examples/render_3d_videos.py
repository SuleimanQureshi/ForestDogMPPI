#!/usr/bin/env python3
"""
Render 3D MuJoCo simulation videos for all ablation study cases.

For each case folder in ablation_results/, loads the saved replay logs
and renders a simulation_3d.mp4 using MuJoCo's offscreen renderer piped
through ffmpeg. No display required.

Usage:
    python examples/render_3d_videos.py                          # All cases
    python examples/render_3d_videos.py --case case0_full_system # Single case
    python examples/render_3d_videos.py --output-dir ablation_results
    python examples/render_3d_videos.py --force                  # Re-render existing
    python examples/render_3d_videos.py --width 1280 --height 720
    python examples/render_3d_videos.py --fps 60
"""
import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mujoco as mj

REPO = Path(__file__).resolve().parent.parent
FFMPEG = "/home/suleiman/miniconda3/envs/go2-convex-mpc/bin/ffmpeg"
def render_case(case_dir: str, xml_path: str, output_path: str,
                width=1280, height=720, fps=60):
    """
    Render simulation_3d.mp4 for one ablation case from saved .npy logs.

    Args:
        case_dir:    path to the case folder (contains *.npy files)
        xml_path:    path to the Mujoco scene XML used
        output_path: full path for the output .mp4
        width/height: video resolution
        fps:         output frame rate
    """
    case_dir = Path(case_dir)
    time_log = np.load(case_dir / "time_log_render.npy")
    q_log    = np.load(case_dir / "q_log_render.npy")

    T = len(time_log)
    print(f"  Sim time: {time_log[-1]:.1f}s")

    # ── Load MuJoCo model ──
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)

    base_id = model.body("base_link").id

    # ── Expand offscreen framebuffer to match requested resolution ──
    model.vis.global_.offwidth  = width
    model.vis.global_.offheight = height

    # ── Offscreen renderer ──
    renderer = mj.Renderer(model, height=height, width=width)

    # Configure tracking camera via MjvCamera (MuJoCo 3.x API)
    cam = mj.MjvCamera()
    cam.type        = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base_id
    cam.fixedcamid  = -1
    cam.distance    = 3.0
    cam.elevation   = -20.0
    cam.azimuth     = 90.0

    # ── FFmpeg process (raw RGB24 stdin → mp4) ──
    cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-preset", "fast",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

    t0 = time.perf_counter()
    next_render_time = time_log[0]
    dt_video = 1.0 / fps
    frames_written = 0
    try:
        for k in range(T):
            if time_log[k] < next_render_time and k < T - 1:
                continue
            next_render_time += dt_video
            frames_written += 1

            data.qpos[:] = q_log[k]
            mj.mj_forward(model, data)

            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()           # (H, W, 3) uint8 RGB
            proc.stdin.write(pixels.tobytes())

            if frames_written % 100 == 0:
                elapsed = time.perf_counter() - t0
                fps_so_far = frames_written / elapsed
                print(f"    {frames_written} frames  "
                      f"({fps_so_far:.0f} fps render)",
                      flush=True)

        proc.stdin.close()
        proc.wait()
    except Exception as e:
        proc.stdin.close()
        proc.kill()
        raise e

    renderer.close()
    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Done in {elapsed:.1f}s  →  {output_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Render 3D MuJoCo videos for ablation study cases")
    parser.add_argument("--case", type=str,
                        help="Render a single case by folder name")
    parser.add_argument("--output-dir", type=str, default="ablation_results",
                        help="Top-level results directory (default: ablation_results)")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if simulation_3d.mp4 already exists")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps",    type=int, default=60)
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    if not results_dir.exists():
        print(f"Output directory not found: {results_dir}")
        sys.exit(1)

    # Collect case directories
    if args.case:
        case_dirs = [results_dir / args.case]
    else:
        case_dirs = sorted(
            d for d in results_dir.iterdir()
            if d.is_dir() and (d / "q_log_render.npy").exists()
        )

    if not case_dirs:
        print("No case directories found.")
        sys.exit(1)

    print(f"Rendering {len(case_dirs)} case(s) at {args.width}x{args.height} @ {args.fps}fps\n")

    XML_PATH = str(REPO / "models" / "MJCF" / "go2" / "scene_test_forest_updated.xml")

    for i, case_dir in enumerate(case_dirs):
        output_path = str(case_dir / "simulation_3d.mp4")
        print(f"[{i + 1}/{len(case_dirs)}] {case_dir.name}")
        
        env_xml = REPO / "models" / "MJCF" / "go2" / "environments" / f"{case_dir.name}.xml"
        if env_xml.exists():
            xml_path = str(env_xml)
        else:
            xml_path = XML_PATH

        if not args.force and os.path.exists(output_path):
            print(f"  SKIP (simulation_3d.mp4 already exists). Use --force to re-render.")
            continue

        if not (case_dir / "q_log_render.npy").exists():
            print(f"  SKIP (q_log_render.npy not found)")
            continue

        try:
            render_case(str(case_dir), xml_path, output_path,
                        width=args.width, height=args.height, fps=args.fps)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
