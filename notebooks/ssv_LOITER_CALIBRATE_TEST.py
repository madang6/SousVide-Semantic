import os
import time
import statistics
import json

import numpy as np
import imageio
import cv2

import sousvide.flight.vision_preprocess_alternate as vp
# import sousvide.flight.zed_command_helper as zed

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
    prompt = cfg.get('prompt', '')
    hf_model = cfg.get('hf_model', 'CIDAS/clipseg-rd64-refined')
    onnx_model_path = cfg.get('onnx_model_path')
    camera_mode = cfg.get('camera_mode', False)

    # Initialize CLIPSeg model (may export & load ONNX)
    if onnx_model_path is None:
        print("Initializing CLIPSegHFModel...")
        model = vp.CLIPSegHFModel(hf_model=hf_model)
    else:
        print("Initializing ONNX CLIPSegHFModel (this may export ONNX)...")
        model = vp.CLIPSegHFModel(
            hf_model=hf_model,
            onnx_model_path=onnx_model_path,
            onnx_model_fp16_path=cfg.get('onnx_model_fp16_path', None)
        )

    times = []
    frames = []
    frame_count = 0

    # ——— File-based video benchmarking via imageio[ffmpeg] ———
    input_video_path = cfg['input_video_path']
    video_dir = os.path.dirname(input_video_path)
    base, _ = os.path.splitext(os.path.basename(input_video_path))
    suffix = "_onnx_benchmark" if onnx_model_path else "_default_benchmark"
    output_path = os.path.join(video_dir, f"{base}{suffix}.mp4")
    output_img_path = os.path.join(video_dir, f"{base}{suffix}_overlay.png")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an imageio writer for H.264-encoded MP4 with yuv420p pixel format
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        # ffmpeg_params=['-pix_fmt', 'yuv420p'],
        # output_params=[]
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames at {fps:.2f} FPS…")

    times, frame_count = [], 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert to RGB for your inference call
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # frame_rgb = frame_bgr

        t10 = time.time()
        overlay, scaled = model.clipseg_hf_inference(
            frame_rgb,
            prompt,
            resize_output_to_input=True,
            use_refinement=False,
            use_smoothing=False,
            scene_change_threshold=1.0,
            verbose=False,
        )
        video_time_elapsed = (frame_count - 0) / fps
        if video_time_elapsed < 3.5:
            found, sim_score, area_frac, overlay = model.loiter_calibrate(
                logits=scaled,           # your logits/similarity map
                frame_img=frame_rgb,       # original frame in RGB
                active_arm=False
            )
            # found, sim_score, area_frac = model.loiter_calibrate(logits=scaled,
            #                                                     active_arm=False)
        else:
            found, sim_score, area_frac, overlay = model.loiter_calibrate(
                logits=scaled,           # your logits/similarity map
                frame_img=frame_rgb,         # original frame in BGR
                active_arm=True
            )
            # found, sim_score, area_frac = model.loiter_calibrate(logits=scaled,
            #                                                     active_arm=True)
            if found:
                break
                
        t11 = time.time()

        times.append(t11 - t10)
        frame_count += 1

        # imageio expects an RGB array
        # writer.append_data(scaled.astype(np.uint8))
        writer.append_data(overlay.astype(np.uint8))

        if frame_count % 50 == 0:
            avg_ms = statistics.mean(times[-50:]) * 1e3
            print(f"  Frame {frame_count}/{total_frames}: avg {avg_ms:.1f} ms/frame")
    
    # save best mask
    overlay = model._make_overlay(frame_rgb, model.loiter_mask)
    cv2.imwrite(output_img_path, overlay)
    # print stats
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

    # Print timing stats
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