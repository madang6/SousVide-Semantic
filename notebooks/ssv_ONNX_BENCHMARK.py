#!/usr/bin/env python3
"""
Benchmark ONNX vs PyTorch CLIPSeg inference on a video.

Configuration is loaded from a JSON file at CONFIG_PATH.
"""
import os
import time
import statistics
import json

import cv2

import sousvide.flight.vision_preprocess_alternate as vp

# Path to the external JSON config file
CONFIG_PATH = (
    "/home/admin/StanfordMSL/SousVide-Semantic/"
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


def main():
    # Load user configuration
    cfg = load_config(CONFIG_PATH)
    input_video_path = cfg['input_video_path']
    prompt = cfg.get('prompt', '')
    hf_model = cfg.get('hf_model', 'CIDAS/clipseg-rd64-refined')
    onnx_model_path = cfg.get('onnx_model_path')

    # Derive output video path
    video_dir = os.path.dirname(input_video_path)
    base, ext = os.path.splitext(os.path.basename(input_video_path))
    output_path = os.path.join(video_dir, f"{base}_onnx_benchmark{ext}")

    # Initialize CLIPSeg model
    print("Initializing CLIPSegHFModel (this may export ONNX)...")
    model = vp.CLIPSegHFModel(
        hf_model=hf_model,
        onnx_model_path=onnx_model_path
    )

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames at {fps:.2f} FPS...")

    times = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # BGR→RGB for model
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Inference
        t0 = time.time()
        overlay = model.clipseg_hf_inference(
            frame_rgb,
            prompt,
            resize_output_to_input=True,
            use_refinement=False,
            scene_change_threshold=1.0,
            verbose=False,
        )
        t1 = time.time()
        times.append(t1 - t0)

        # Write output frame (convert back to BGR if RGB)
        if overlay.ndim == 3 and overlay.shape[2] == 3:
            out_frame = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        else:
            out_frame = overlay
        out.write(out_frame)

        frame_idx += 1
        if frame_idx % 50 == 0:
            avg_ms = statistics.mean(times[-50:]) * 1000
            print(f"  Frame {frame_idx}/{total_frames}  avg {avg_ms:.1f} ms/frame")

    cap.release()
    out.release()

    # Print timing stats
    print("Benchmark completed.")
    print(f"Output video: {output_path}")
    print(f"Total frames: {frame_idx}")
    total_time = sum(times)
    print(f"Total inference time: {total_time:.2f} s")
    print(f"Average time/frame: {statistics.mean(times)*1000:.1f} ms")
    print(f"Median time/frame: {statistics.median(times)*1000:.1f} ms")
    print(f"Min time/frame: {min(times)*1000:.1f} ms")
    print(f"Max time/frame: {max(times)*1000:.1f} ms")


if __name__ == "__main__":
    main()