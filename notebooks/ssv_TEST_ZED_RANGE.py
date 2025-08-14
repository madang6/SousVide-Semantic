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


def ensure_hw_match(reference_hw: tuple[int, int], arr: np.ndarray) -> np.ndarray:
    H, W = reference_hw
    if arr.shape[:2] != (H, W):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
    return arr


# ---------- Main script ----------

def parse_args():
    p = argparse.ArgumentParser()
    # Camera init
    p.add_argument("--cam-width",  type=int, default=None)
    p.add_argument("--cam-height", type=int, default=None)
    p.add_argument("--cam-fps",    type=int, default=None)
    # CLIPSeg
    p.add_argument("--prompt", type=str, default="boxes")
    p.add_argument("--hf-model", type=str, default="CIDAS/clipseg-rd64-refined")
    p.add_argument("--onnx-model-path", type=str, default=None)
    p.add_argument("--onnx-model-fp16-path", type=str, default=None)
    p.add_argument("--mode", choices=["full", "depth_only"], default="full",
                   help="full = CLIPSeg+range+3 videos, depth_only = capture XYZ+VIEW.DEPTH, print center range, save depth video only")
    p.add_argument("--range-source", choices=["xyz", "z"], default="xyz",
                   help="For depth_only: use 'xyz' for Euclidean range (MEASURE.XYZ) or 'z' for optical-axis depth (MEASURE.DEPTH)")
    # Estimator
    p.add_argument("--top-percent", type=float, default=10.0)
    p.add_argument("--min-pixels",  type=int, default=200)
    p.add_argument("--z-min",  type=float, default=0.3)
    p.add_argument("--z-max",  type=float, default=20.0)
    # Saving
    p.add_argument("--input-video-path", dest="input_video_path", type=str, default=None)
    p.add_argument("--out-clipseg", type=str, default="clipseg_out.mp4")
    p.add_argument("--out-depth",   type=str, default="depth_out.mp4")
    p.add_argument("--out-sideby",  type=str, default="combo_out.mp4")
    p.add_argument("--save-fps",    type=float, default=15.0)
    # Loop
    p.add_argument("--rate-hz",     type=float, default=15.0)   # encode rate & print throttle
    p.add_argument("--max-seconds", type=float, default=None)   #
    return p.parse_args()

def pick(val, cfg, key, default=None):
    return val if val is not None else cfg.get(key, default)


def main():
    args = parse_args()

    # Load JSON config (same path you already use)
    CONFIG_PATH = os.path.expanduser(
        "~/StanfordMSL/SousVide-Semantic/configs/perception/onnx_benchmark_config.json"
    )
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    # Merge settings (CLI > cfg > hard default)
    mode          = pick(args.mode,          cfg, "mode",           "full")
    range_source  = pick(args.range_source,  cfg, "range_source",   "xyz")
    width         = pick(args.cam_width,  cfg, "camera_width",   640)
    height        = pick(args.cam_height, cfg, "camera_height",  480)
    fps_cam       = pick(args.cam_fps,    cfg, "camera_fps",     30)
    duration      = pick(args.max_seconds, cfg, "camera_duration", 10.0)
    save_fps      = pick(args.save_fps,      cfg, "save_fps",       10.0)

    input_video_path = pick(args.input_video_path, cfg, "input_video_path",
                            "~/StanfordMSL/SousVide-Semantic/notebooks/out/live_placeholder.mp4")
    input_video_path = os.path.expanduser(input_video_path)
    video_dir = os.path.dirname(input_video_path)
    _, ext = os.path.splitext(os.path.basename(input_video_path))
    out_clipseg = os.path.join(video_dir, f"live_range_clipseg{ext}")
    out_depth   = os.path.join(video_dir, f"live_range_depth{ext}")
    out_combo   = os.path.join(video_dir, f"live_range_combo{ext}")

    # Open camera (ensure depth is enabled in your get_camera)
    cam = zch.get_camera(height=height, width=width, fps=fps_cam, use_depth=True)
    if cam is None:
        raise RuntimeError("Unable to initialize ZED camera.")

# ---------- DEPTH-ONLY MODE ----------

    if mode == "depth_only":
        print(f"Running depth_only mode (range-source={range_source})")
        frames_depth = []
        frame_count = 0
        t0 = time.time()
        
    # ---------- CLIPSeg Noise Test ----------
        prompt = cfg.get("prompt", "")
        print(f"Prompt set to {prompt}")
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
    # ---------- END CLIPSeg Noise Test ----------

        try:
            while (time.time() - t0) < float(duration):
                # Your get_image returns: image, XYZ (or None), VIEW.DEPTH display, ts_ms
                img_np, xyz_np, depth_viz, t_ms = zch.get_image(cam, use_depth=True)
                if depth_viz is None:
                    continue

            # ---------- CLIPSeg Noise Test ----------
                # ZED often returns BGRA for VIEW.LEFT; convert to RGB for the model
                img_bgr = img_np[..., :3] if img_np.shape[2] >= 3 else img_np
                frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                out = model.clipseg_hf_inference(
                    frame_rgb,
                    prompt,
                    resize_output_to_input=True,
                    use_refinement=False,
                    use_smoothing=False,
                    scene_change_threshold=1.0,
                    verbose=False,
                )
            # ---------- END CLIPSeg Noise Test ----------

                # Depth display → RGB for imageio
                H, W = depth_viz.shape[:2]
                depth_rgb = vp.depth_display_to_rgb(depth_viz, target_hw=(H, W))

                # Center-pixel range readout
                cy, cx = H // 2, W // 2
                center_txt = "N/A"
                if range_source == "xyz" and isinstance(xyz_np, np.ndarray) and xyz_np.ndim == 3 and xyz_np.shape[2] >= 3:
                    p = xyz_np[cy, cx, :3].astype(np.float64, copy=False)
                    if np.all(np.isfinite(p)):
                        r = float(np.hypot(np.hypot(p[0], p[1]), p[2]))
                        center_txt = f"{r:.2f} m"
                elif range_source == "z":
                    # If get_image is returning XYZ, use its Z channel; if you switch it back to MEASURE.DEPTH (HxW),
                    # this still works (ndim==2 path)
                    if isinstance(xyz_np, np.ndarray):
                        z = xyz_np[cy, cx, 2] if xyz_np.ndim == 3 else xyz_np[cy, cx]
                        if np.isfinite(z) and z > 0:
                            center_txt = f"{float(z):.2f} m"

                # Annotate & collect
                depth_rgb = depth_rgb.copy()
                cv2.putText(depth_rgb, f"Center range ({range_source}): {center_txt}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,10,10), 3, cv2.LINE_AA)
                cv2.putText(depth_rgb, f"Center range ({range_source}): {center_txt}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 1, cv2.LINE_AA)

                frames_depth.append(depth_rgb)
                frame_count += 1

                # pace capture around save_fps
                time.sleep(max(0.0, 1.0/float(save_fps) - 0.001))

        finally:
            zch.close_camera(cam)
            if frame_count > 0:
                elapsed = max(1e-6, time.time() - t0)
                avg_fps = frame_count / elapsed
                imageio.mimsave(out_depth, frames_depth, fps=avg_fps)
                print(f"[depth_only] Frames: {frame_count}  avg_fps≈{avg_fps:.2f}")
                print(f"[depth_only] Wrote depth video: {out_depth}")
            else:
                print("[depth_only] No frames captured; nothing to save.")
        return
# ---------- END DEPTH-ONLY MODE ----------

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

    # Range estimator params
    top_percent = float(cfg.get("range_top_percent", 10.0))
    min_pixels  = int(cfg.get("range_min_pixels", 200))

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
            img_np, xyz_np, depth_viz, t_ms = zch.get_image(cam, use_depth=True)
            if img_np is None or xyz_np is None or depth_viz is None:
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
            xyz_np = ensure_hw_match((H, W), xyz_np)

            # Estimate range (top-P% similarity pixels)
            ok, range_m = zch.euclidean_range_from_similarity(
                similarity, xyz_np,
                top_percent=top_percent,
                min_pixels=min_pixels,
            )

            # Build frames for saving (imageio expects RGB)
            clipseg_rgb = overlay_rgb

            # Convert ZED VIEW.DEPTH (8-bit display) to RGB for saving, and match overlay size
            H, W = clipseg_rgb.shape[:2]
            depth_rgb = vp.depth_display_to_rgb(depth_viz, target_hw=(H, W))

            # Optional label
            # depth_rgb = depth_rgb.copy()
            # label = f"Depth (display)  Range est: {range_m:.2f} m" if ok and np.isfinite(range_m) else "Depth (display)  Range est: N/A"
            # cv2.putText(depth_rgb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,10,10), 3, cv2.LINE_AA)
            # cv2.putText(depth_rgb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 1, cv2.LINE_AA)

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
