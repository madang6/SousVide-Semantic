#!/usr/bin/env python3
"""
ssv_TEST_RANGE.py
- Opens ZED via zed_command_helper.get_camera(...)
- Grabs LEFT image + DEPTH (meters) using zed_command_helper.get_image(..., use_depth=True)
- Runs CLIPSeg to get similarity map
- Estimates range over the top-P% similarity pixels
- Saves three MP4s: CLIPSeg overlay, depth (colorized), side-by-side
"""
import os
import json

import argparse
import time
from typing import Tuple, Optional

import numpy as np
import cv2
import imageio

# Project modules
import sousvide.flight.vision_preprocess_alternate as vp
import sousvide.flight.zed_command_helper as zch



# Path to the external JSON config file
CONFIG_PATH_RAW = (
    "~/StanfordMSL/SousVide-Semantic/"
    "configs/perception/onnx_benchmark_config.json"
)



def load_config(path):
    """
    Load benchmark configuration from JSON.
    Expected keys:
      - input_video_path: str
      - prompt: str
      - hf_model: str (HuggingFace model name)
      - onnx_model_path: str or null
    """
    with open(path, 'r') as f:
        return json.load(f)


# ---------- Viz helpers ----------

def colorize_depth(depth_m: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """
    Convert a depth image (meters, NaN/inf invalid) to a BGR colormap.
    Invalid pixels are rendered black. No RuntimeWarnings.
    """
    d = depth_m.astype(np.float32, copy=False)

    # Valid mask BEFORE any transforms
    valid = np.isfinite(d) & (d > 0.0)

    # Guard denom
    denom = float(z_max - z_min)
    if denom <= 0 or not np.isfinite(denom):
        denom = 1.0

    # Build a safe array for normalization: default to z_min, fill valid with clipped depths
    d_safe = np.full_like(d, z_min, dtype=np.float32)
    if np.any(valid):
        d_safe[valid] = np.clip(d[valid], z_min, z_max)

    # Normalize 0..1 only on the safe array (no NaNs/Infs remain)
    d_norm = (d_safe - z_min) / denom
    d_norm = np.clip(d_norm, 0.0, 1.0)

    # 0..255 u8 without warnings (+0.5 for round-to-nearest)
    d_u8 = (d_norm * 255.0 + 0.5).astype(np.uint8)

    # Apply colormap and paint invalid as black
    cm = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    cm[~valid] = (0, 0, 0)
    return cm


def ensure_hw_match(reference_hw: tuple[int, int], arr: np.ndarray) -> np.ndarray:
    H, W = reference_hw
    if arr.shape[:2] != (H, W):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
    return arr


# ---------- Main script ----------

def parse_args():
    p = argparse.ArgumentParser("ZED SDK test: CLIPSeg similarity → depth range; save videos.")
    # Camera init
    p.add_argument("--cam-width",  type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps",    type=int, default=30)
    # CLIPSeg
    p.add_argument("--prompt", type=str, default="boxes")
    p.add_argument("--hf-model", type=str, default="CIDAS/clipseg-rd64-refined")
    p.add_argument("--onnx-model-path", type=str, default=None)
    p.add_argument("--onnx-model-fp16-path", type=str, default=None)
    # Estimator
    p.add_argument("--top-percent", type=float, default=10.0)
    p.add_argument("--min-pixels",  type=int, default=200)
    p.add_argument("--z-min",  type=float, default=0.3)
    p.add_argument("--z-max",  type=float, default=20.0)
    # Saving
    p.add_argument("--out-clipseg", type=str, default="clipseg_out.mp4")
    p.add_argument("--out-depth",   type=str, default="depth_out.mp4")
    p.add_argument("--out-sideby",  type=str, default="combo_out.mp4")
    p.add_argument("--save-fps",    type=float, default=15.0)
    # Loop
    p.add_argument("--rate-hz",     type=float, default=15.0)  # encode rate & print throttle
    p.add_argument("--max-seconds", type=float, default=10.0)   # 0 = unlimited
    return p.parse_args()


def main():
    # Load config like the benchmark
    CONFIG_PATH = os.path.expanduser(CONFIG_PATH_RAW)
    cfg = load_config(CONFIG_PATH)

    # Where to save (same logic as benchmark live path)
    input_video_path = cfg["input_video_path"]                      
    video_dir = os.path.dirname(input_video_path)                   
    _, ext = os.path.splitext(os.path.basename(input_video_path))   
    out_clipseg = os.path.join(video_dir, f"live_range_clipseg{ext}")
    out_depth   = os.path.join(video_dir, f"live_range_depth{ext}")
    out_combo   = os.path.join(video_dir, f"live_range_combo{ext}")

    # Model selection (same knobs as benchmark)
    prompt = cfg.get("prompt", "")
    hf_model = cfg.get("hf_model", "CIDAS/clipseg-rd64-refined")
    onnx_model_path = cfg.get("onnx_model_path")
    if onnx_model_path is None:
        print("Initializing CLIPSegHFModel (PyTorch)…")
        model = vp.CLIPSegHFModel(hf_model=hf_model)               
    else:
        onnx_model_path = os.path.expanduser(onnx_model_path)
        print("Initializing CLIPSegHFModel (ONNX)…")
        model = vp.CLIPSegHFModel(                                 
            hf_model=hf_model,
            onnx_model_path=onnx_model_path,
            onnx_model_fp16_path=cfg.get("onnx_model_fp16_path", None),
        )

    # Camera knobs (mirror ‘camera_*’ keys the benchmark uses)
    fps_cam  = cfg.get("camera_fps",    30)                        
    width    = cfg.get("camera_width",  640)                       
    height   = cfg.get("camera_height", 480)                       
    duration = cfg.get("camera_duration", 10.0)                    

    # Range estimator params
    top_percent = float(cfg.get("range_top_percent", 10.0))
    min_pixels  = int(cfg.get("range_min_pixels", 200))
    z_min       = float(cfg.get("range_z_min", 0.3))
    z_max       = float(cfg.get("range_z_max", 20.0))

    # Open camera (same helper)
    cam = zch.get_camera(height=height, width=width, fps=fps_cam)
    if cam is None:
        raise RuntimeError("Unable to initialize ZED camera.")

    frames_clipseg = []
    frames_depth   = []
    frames_combo   = []
    times = []
    frame_count = 0

    print(f"Capturing live for {duration:.1f}s at {fps_cam} FPS…")
    t_start = time.time()

    try:
        while (time.time() - t_start) < duration:
            # Grab LEFT + DEPTH (meters) with your updated get_image
            img_np, depth_np, depth_viz, t_ms = zch.get_image(cam, use_depth=True)
            if img_np is None or depth_np is None:
                continue

            # ZED often returns BGRA for VIEW.LEFT; convert to RGB for the model
            img_bgr = img_np[..., :3] if img_np.shape[2] >= 3 else img_np
            frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # CLIPSeg inference → overlay + similarity
            t0 = time.time()
            out = model.clipseg_hf_inference(
                frame_rgb,
                prompt,
                resize_output_to_input=True,
                use_refinement=False,
                use_smoothing=False,
                scene_change_threshold=1.0,
                verbose=False,
            )
            t1 = time.time()
            times.append(t1 - t0)

            if isinstance(out, tuple) and len(out) >= 2:
                overlay_rgb, similarity = out[0], out[1]
            else:
                overlay_rgb = out
                similarity = getattr(model, "latest_similarity", None)
            if similarity is None:
                continue

            # Make sure depth matches overlay size (nearest)
            H, W = overlay_rgb.shape[:2]
            depth_np = ensure_hw_match((H, W), depth_np)

            # Estimate range (top-P% similarity pixels)
            ok, range_m = zch.euclidean_range_from_similarity(
                similarity, depth_np,
                top_percent=top_percent,
                min_pixels=min_pixels,
            )

            # Build frames for saving (imageio expects RGB)
            clipseg_rgb = overlay_rgb

            # Convert ZED VIEW.DEPTH (8-bit display) to RGB for saving, and match overlay size
            H, W = clipseg_rgb.shape[:2]
            depth_rgb = vp.depth_display_to_rgb(depth_viz, target_hw=(H, W))

            # Optional label
            depth_rgb = depth_rgb.copy()
            label = f"Depth (display)  Range est: {range_m:.2f} m" if ok and np.isfinite(range_m) else "Depth (display)  Range est: N/A"
            cv2.putText(depth_rgb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,10,10), 3, cv2.LINE_AA)
            cv2.putText(depth_rgb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 1, cv2.LINE_AA)

            # Append for imageio
            frames_clipseg.append(clipseg_rgb)
            frames_depth.append(depth_rgb)
            frames_combo.append(np.hstack([clipseg_rgb, depth_rgb]))
            frame_count += 1

            # Optional console
            if frame_count % max(1, int(fps_cam/2)) == 0:
                print(
                    f"[{frame_count:04d}] "
                    f"range(top {top_percent:.1f}%): "
                    f"{range_m:.2f} m" if ok and np.isfinite(range_m) else "[NO RANGE]"
                )

    finally:
        zch.close_camera(cam)                                          
        print("Live capture complete. Saving videos with imageio…")

        # Match benchmark’s save style: compute avg FPS from inference timings and mimsave
        if frame_count > 0:
            total_time = sum(times) if times else max(1e-6, time.time() - t_start)
            avg_fps = frame_count / total_time
            print(f"Frames: {frame_count}  avg_fps≈{avg_fps:.2f}")

            # Save three videos to the SAME directory the benchmark uses
            imageio.mimsave(out_clipseg, frames_clipseg, fps=avg_fps)  
            imageio.mimsave(out_depth,   frames_depth,   fps=avg_fps)  # same style
            imageio.mimsave(out_combo,   frames_combo,   fps=avg_fps)  # side-by-side
            print(f"Wrote:\n  {out_clipseg}\n  {out_depth}\n  {out_combo}")
        else:
            print("No frames captured; nothing to save.")

if __name__ == "__main__":
    main()