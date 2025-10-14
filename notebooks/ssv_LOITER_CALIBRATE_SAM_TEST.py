import os
import time
import statistics
import json

import numpy as np
import imageio
import cv2

import sousvide.flight.vision_preprocess_groundedsam as vp
# import sousvide.flight.zed_command_helper as zed

# Path to the external JSON config file
CONFIG_PATH = (
    "/home/cassie/StanfordMSL/SousVide-Semantic/"
    "configs/perception/onnx_benchmark_dino_config.json"
)


def load_config(path):
    """
    Load benchmark configuration from JSON.
    Expected keys:
      - engine: "clipseg" | "groundedsam"
      - input_video_path: str
      - prompt: str
      - hf_model: str (for CLIPSeg)
      - onnx_model_path: str or null (for CLIPSeg)
      - onnx_model_fp16_path: str or null (optional, for CLIPSeg)
      - camera_mode: bool (optional)
      - gd_model_id: str (for Grounded SAM)
      - sam_model_id: str (for Grounded SAM)
      - box_threshold, text_threshold, mask_iou_threshold, overlay_alpha (optional, Grounded SAM)
    """
    with open(path, 'r') as f:
        return json.load(f)
    
def _mask_from_softmap_area_targeted(soft01: np.ndarray, area_frac: float) -> np.ndarray:
    """
    Build a 0/255 uint8 mask whose area ≈ area_frac by thresholding the quantile of the soft map.
    soft01 is assumed in [0,1].
    """
    area_frac = float(np.clip(area_frac, 1e-4, 0.90))
    q = 1.0 - area_frac
    t = float(np.quantile(soft01, q))
    return (soft01 >= t).astype(np.uint8) * 255

def _apply_mask_rgb(frame_rgb: np.ndarray, mask_u8: np.ndarray, bg_color=(0, 0, 0)) -> np.ndarray:
    """
    Returns an RGB image where pixels outside mask are painted bg_color.
    mask_u8 is 0/255 (HxW).
    """
    if mask_u8.dtype != np.uint8:
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    bg = np.empty_like(frame_rgb, dtype=np.uint8)
    bg[..., 0] = bg_color[0]
    bg[..., 1] = bg_color[1]
    bg[..., 2] = bg_color[2]
    return np.where(mask_u8[..., None] > 0, frame_rgb, bg)



def _simple_overlay(rgb, softmap01, alpha=0.45):
    """Fallback overlay if model._make_overlay isn't available."""
    soft = np.clip(softmap01, 0.0, 1.0)
    heat = (soft * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1.0, heat, alpha, 0.0)


def main():
    # Load user configuration
    cfg = load_config(CONFIG_PATH)
    prompt = cfg.get('prompt', '')
    engine = cfg.get('engine', 'clipseg').lower()
    camera_mode = cfg.get('camera_mode', False)

    # ==========================
    # Initialize the segmentation model
    # ==========================
    if engine == "clipseg":
        hf_model = cfg.get('hf_model', 'CIDAS/clipseg-rd64-refined')
        onnx_model_path = cfg.get('onnx_model_path')
        if onnx_model_path is None:
            print("Initializing CLIPSegHFModel...")
            model = vp.CLIPSegHFModel(hf_model=hf_model)
            suffix_engine = "_clipseg_default"
        else:
            print("Initializing ONNX CLIPSegHFModel (this may export ONNX)...")
            model = vp.CLIPSegHFModel(
                hf_model=hf_model,
                onnx_model_path=onnx_model_path,
                onnx_model_fp16_path=cfg.get('onnx_model_fp16_path', None)
            )
            suffix_engine = "_clipseg_onnx"
    elif engine == "groundedsam":
        print("Initializing GroundedSAMHFModel...")
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

    times = []
    frames = []
    frame_count = 0

    # ——— File-based video benchmarking via imageio[ffmpeg] ———
    input_video_path = cfg['input_video_path']
    video_dir = os.path.dirname(input_video_path)
    base, _ = os.path.splitext(os.path.basename(input_video_path))
    output_path = os.path.join(video_dir, f"{base}{suffix_engine}_benchmark.mp4")
    output_img_path = os.path.join(video_dir, f"{base}{suffix_engine}_overlay.png")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames at {fps:.2f} FPS…")

    times, frame_count = [], 0
    last_overlay = None
    sim_score = 0.0
    area_frac = 0.0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t10 = time.time()

        # ==========================
        # Inference call (engine-specific)
        # ==========================
        if engine == "clipseg":
            overlay, scaled = model.clipseg_hf_inference(
                frame_rgb,
                prompt,
                resize_output_to_input=True,
                use_refinement=False,
                use_smoothing=False,
                scene_change_threshold=1.0,
                verbose=False,
            )
            present = True  # CLIPSeg path historically didn't expose presence; keep behavior consistent
        else:
            overlay, scaled, present, _extras = model.grounded_sam_hf_inference(
                frame_rgb,
                prompt,
                resize_output_to_input=True,
                use_refinement=False,
                use_smoothing=False,
                scene_change_threshold=1.0,
                verbose=False,
            )

        # ==========================
        # Loiter calibration logic (unchanged)
        # ==========================
        video_time_elapsed = (frame_count - 0) / fps
        if video_time_elapsed < 3.5:
            found, sim_score, area_frac, overlay = model.loiter_calibrate(
                logits=scaled,           # soft/prob-like map
                frame_img=frame_rgb,     # original frame in RGB
                active_arm=False
            )
        else:
            found, sim_score, area_frac, overlay = model.loiter_calibrate(
                logits=scaled,
                frame_img=frame_rgb,
                active_arm=True
            )
            if found:
                # Optional: you were breaking here; keep consistent
                break

        t11 = time.time()
        times.append(t11 - t10)
        frame_count += 1

        if area_frac > 0.0:
            mask_u8 = _mask_from_softmap_area_targeted(scaled.astype(np.float32), area_frac)
        else:
            # fallback: simple 90th percentile
            t = float(np.percentile(scaled, 90.0))
            mask_u8 = (scaled >= t).astype(np.uint8) * 255

        masked_rgb = _apply_mask_rgb(frame_rgb, mask_u8, bg_color=(0, 0, 0))  # black background
        last_overlay = masked_rgb  # keep for end-of-run save
        writer.append_data(masked_rgb.astype(np.uint8))
        # last_overlay = overlay
        # writer.append_data(overlay.astype(np.uint8))

        if frame_count % 50 == 0:
            avg_ms = statistics.mean(times[-50:]) * 1e3
            print(f"  Frame {frame_count}/{total_frames}: avg {avg_ms:.1f} ms/frame")
    
    # ==========================
    # Save final overlay frame (best loiter mask)
    # ==========================
    # === Save final masked image instead of overlay ===
    if 'frame_rgb' in locals():
        final_mask = getattr(model, "loiter_mask", None)
        if final_mask is not None:
            masked_final = _apply_mask_rgb(frame_rgb, final_mask, bg_color=(0, 0, 0))
            cv2.imwrite(output_img_path, cv2.cvtColor(masked_final, cv2.COLOR_RGB2BGR))
        else:
            # fallback to the last masked frame we produced
            if 'last_overlay' in locals() and last_overlay is not None:
                cv2.imwrite(output_img_path, cv2.cvtColor(last_overlay, cv2.COLOR_RGB2BGR))
    else:
        print("[WARN] No final frame available to save.")

    # print stats (unchanged)
    print(f"largest_area={model.loiter_area_frac*100:.1f}% "
          f", best scoring area={model.loiter_max:.3f} "
          f", sim_score={sim_score:.3f} "
          f", area_frac={area_frac*100:.1f}% "
          f", sim_score_diff={model.loiter_max - sim_score:.3f} "
          f", area_frac_diff={(model.loiter_area_frac - area_frac)*100:.1f}%")

    cap.release()
    writer.close()

    print(f"Output video saved to: {output_path}")
    print("File-based video benchmark completed.")

    # Timing
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
            imageio.mimsave(output_path, frames, fps=avg_fps)
    else:
        print("No frames were processed.")


if __name__ == "__main__":
    main()