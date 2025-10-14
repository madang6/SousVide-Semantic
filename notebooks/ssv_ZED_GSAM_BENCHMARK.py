"""
Benchmark CLIPSeg (PyTorch or ONNX) OR Grounded SAM inference on a video or live camera.

Configuration is loaded from a JSON file at CONFIG_PATH.
"""
import os
import time
import statistics
import json

import numpy as np
import imageio
import cv2

import sousvide.flight.zed_command_helper as zed   # uncomment if you use camera_mode
import sousvide.flight.vision_preprocess_groundedsam as vp

# Path to the external JSON config file
CONFIG_PATH_RAW = (
    "~/StanfordMSL/SousVide-Semantic/"
    "configs/perception/onnx_benchmark_dino_config.json"
)


def load_config(path):
    """
    Load benchmark configuration from JSON.
    Expected keys (examples):
      - engine: "clipseg" | "groundedsam"
      - input_video_path: str
      - prompt: str
      - camera_mode: bool
      # CLIPSeg:
      - hf_model: str
      - onnx_model_path: str or null
      - onnx_model_fp16_path: str or null
      # Grounded SAM:
      - gd_model_id: str
      - sam_model_id: str
      - box_threshold: float
      - text_threshold: float
      - mask_iou_threshold: float
      - overlay_alpha: float
    """
    with open(path, 'r') as f:
        return json.load(f)


def main():
    # Load user configuration
    CONFIG_PATH = os.path.expanduser(CONFIG_PATH_RAW)
    cfg = load_config(CONFIG_PATH)

    engine = cfg.get("engine", "clipseg").lower()
    prompt = cfg.get("prompt", "")
    camera_mode = bool(cfg.get("camera_mode", False))

    # -------------------------------
    # Initialize the selected engine
    # -------------------------------
    if engine == "clipseg":
        hf_model = cfg.get('hf_model', 'CIDAS/clipseg-rd64-refined')
        onnx_model_path_raw = cfg.get('onnx_model_path', None)
        onnx_model_path = os.path.expanduser(onnx_model_path_raw) if onnx_model_path_raw else None

        if onnx_model_path is None:
            print("[Init] CLIPSegHFModel (PyTorch)")
            model = vp.CLIPSegHFModel(hf_model=hf_model)
            suffix_engine = "_clipseg_pt"
        else:
            print("[Init] CLIPSegHFModel (ONNX) – may export ONNX on first run")
            model = vp.CLIPSegHFModel(
                hf_model=hf_model,
                onnx_model_path=onnx_model_path,
                onnx_model_fp16_path=cfg.get('onnx_model_fp16_path', None),
            )
            suffix_engine = "_clipseg_onnx"

    elif engine == "groundedsam":
        print("[Init] GroundedSAMHFModel")
        gd_model_id = cfg.get('gd_model_id', 'IDEA-Research/grounding-dino-tiny')
        sam_model_id = cfg.get('sam_model_id', 'facebook/sam-vit-base')
        box_thr = float(cfg.get('box_threshold', 0.35))
        text_thr = float(cfg.get('text_threshold', 0.25))
        mask_iou_thr = float(cfg.get('mask_iou_threshold', 0.0))
        overlay_alpha = float(cfg.get('overlay_alpha', 0.45))

        model = vp.GroundedSAMHFModel(
            gd_model_id=gd_model_id,
            sam_model_id=sam_model_id,
            box_threshold=box_thr,
            text_threshold=text_thr,
            mask_iou_threshold=mask_iou_thr,
            overlay_alpha=overlay_alpha,
            return_patches=False,
        )
        suffix_engine = "_groundedsam"

    else:
        raise ValueError(f"Unknown engine='{engine}'. Use 'clipseg' or 'groundedsam'.")

    times: list[float] = []
    frames: list[np.ndarray] = []
    frame_count = 0

    # ---------------------------------------------------
    # Live camera benchmarking (optional / off by default)
    # ---------------------------------------------------
    if camera_mode:
        # NOTE: Uncomment zed imports and use your camera helpers if needed
        input_video_path = cfg.get('input_video_path', '/tmp/live.mp4')  # for naming only
        video_dir = os.path.dirname(input_video_path)
        base, ext = os.path.splitext(os.path.basename(input_video_path))
        output_path = os.path.join(video_dir, f"live_{suffix_engine}_benchmark{ext or '.mp4'}")

        fps_cam = cfg.get('camera_fps', 30)
        duration = cfg.get('camera_duration', 10.0)
        width = cfg.get('camera_width', 640)
        height = cfg.get('camera_height', 480)

        camera = zed.get_camera(height=height, width=width, fps=fps_cam)
        if camera is None:
            raise RuntimeError("Unable to initialize camera.")
        print(f"Capturing live for {duration:.1f}s at {fps_cam} FPS...")
        start_time = time.time()

        while (time.time() - start_time) < duration:
            frame, _, _, timestamp = zed.get_image(camera)
            if frame is None: continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            if engine == "clipseg":
                overlay, _ = model.clipseg_hf_inference(
                    frame_rgb, prompt, True, False, False, 1.0, False
                )
            else:
                overlay, _, _, _ = model.grounded_sam_hf_inference(
                    frame_rgb, prompt, True, False, False, 1.0, False
                )
            t1 = time.time()
            times.append(t1 - t0)
            frames.append(overlay)
            frame_count += 1

        zed.close_camera(camera)
        print("Live camera benchmarking completed. Saving processed frames...")
        # print("[WARN] camera_mode=True stub left in place; plug in your ZED calls if needed.")
        # return  # remove this return once you wire camera back in
    
    else:
        # ------------------------
        # File-based benchmarking
        # ------------------------
        input_video_path = cfg['input_video_path']
        video_dir = os.path.dirname(input_video_path)
        base, ext = os.path.splitext(os.path.basename(input_video_path))
        output_path = os.path.join(video_dir, f"{base}{suffix_engine}_benchmark.mp4")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {total_frames} frames at {fps:.2f} FPS…")

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            if engine == "clipseg":
                # clipseg_hf_inference returns (overlayed_rgb, scaled_softmap)
                overlay, _scaled = model.clipseg_hf_inference(
                    frame_rgb,
                    prompt,
                    resize_output_to_input=True,
                    use_refinement=False,
                    use_smoothing=False,
                    scene_change_threshold=1.0,
                    verbose=False,
                )
            else:
                # grounded_sam_hf_inference returns (overlayed_rgb, scaled_softmap, present_flag, extras)
                overlay, _scaled, _present, _extras = model.grounded_sam_hf_inference(
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
            frame_count += 1

            # write RGB → BGR
            out.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if frame_count % 50 == 0:
                avg_ms = statistics.mean(times[-50:]) * 1000
                print(f"  Frame {frame_count}/{total_frames}  avg {avg_ms:.1f} ms/frame")

        cap.release()
        out.release()
        print(f"Output video: {output_path}")
        print("File-based video benchmark completed.")

    # ------------------------
    # Timing stats
    # ------------------------
    if frame_count > 0:
        total_time = sum(times)
        avg_fps = frame_count / total_time
        print(f"Total frames processed: {frame_count}")
        print(f"Total inference time: {total_time:.2f} s")
        print(f"Average time/frame: {statistics.mean(times)*1000:.1f} ms")
        print(f"Median time/frame: {statistics.median(times)*1000:.1f} ms")
        print(f"Min time/frame: {min(times)*1000:.1f} ms")
        print(f"Max time/frame: {max(times)*1000:.1f} ms")
        print(f"Average FPS: {avg_fps:.2f}")

        if frames:
            # If you used the live camera branch with frames accumulation:
            imageio.mimsave(output_path, frames, fps=avg_fps)
            print(f"Output video saved to: {output_path}")
    else:
        print("No frames were processed.")


if __name__ == "__main__":
    main()
