#%%
import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim
from scipy.stats import mode

import torch
from torch import nn
import torch.nn.functional as F
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")
import onnxruntime as ort

import cv2
import imageio
from PIL import Image
import albumentations as A
from einops import rearrange
import gc

from abc import abstractmethod, abstractproperty
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type, Union

import sousvide.flight.zed_command_helper as ch
import sousvide.flight.vision_preprocess as vp

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # Disable Albumentations update check

try:
    import figs.render.gsplat_semantic as nf
except:
    print("Unable to load NeRF/3DGS utilities. This is expected if you are not using nerfstudio.")

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
except:
    print("Unable to load CLIPSeg. This is expected if you are not using CLIPSeg.")



class CLIPSegONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values, input_ids, attention_mask):
        return self.model(pixel_values=pixel_values,
                          input_ids=input_ids,
                          attention_mask=attention_mask).logits

def export_onnx_model(
    onnx_path,
    dummy_image,dummy_text,
    processor,model,device
    ):

    model.eval()
    dummy_image = preprocess_vga_image(dummy_image)
    inputs = processor(images=dummy_image, 
                       text=dummy_text, 
                       return_tensors="pt")
    
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2. Wrap model
    wrapped_model = CLIPSegONNXWrapper(model).to(device)
    wrapped_model.eval()

    # Export ONNX
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (pixel_values, input_ids, attention_mask),
            onnx_path,
            input_names=["pixel_values", "input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch", 2: "height", 3: "width"},
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "height", 2: "width"},
            },
            opset_version=13,
            do_constant_folding=True
        )
    print(f"[INFO] ONNX model successfully exported to: {onnx_path}")

# def clipseg_inference_onnx(
#     image: Image.Image,
#     prompt: str,
#     processor,
#     session: ort.InferenceSession,
#     resize_output_to_input: bool = True
# ) -> Image.Image:
#     """
#     Run ONNX Runtime inference for CLIPSeg using an ONNX model.
#     Returns a PIL Image mask (0-255).
#     """

#     # 1. Preprocess input using the same processor
#     inputs = processor(images=image, text=prompt, return_tensors="pt")
#     pixel_values = inputs["pixel_values"].numpy()
#     input_ids = inputs["input_ids"].numpy()
#     attention_mask = inputs["attention_mask"].numpy()

#     # 2. Run ONNX inference
#     ort_inputs = {
#         "pixel_values": pixel_values,
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#     }
#     ort_outputs = session.run(["logits"], ort_inputs)
#     prob_map = ort_outputs[0][0]  # shape: (1, H, W)

#     # 3. Normalize to 0–255 and convert to PIL image
#     min_val = np.min(prob_map)
#     max_val = np.max(prob_map)
#     if max_val - min_val < 1e-9:
#         scaled = np.zeros_like(prob_map, dtype=np.float32)
#     else:
#         scaled = (prob_map - min_val) / (max_val - min_val)
#     mask_pil = Image.fromarray((scaled * 255).astype(np.uint8))

#     # 4. Resize to input size if needed
#     if resize_output_to_input:
#         mask_pil = mask_pil.resize(image.size, resample=Image.BILINEAR)

#     return mask_pil
def clipseg_inference_onnx_gpu(
    image: Image.Image,
    prompt: str,
    processor,
    session: ort.InferenceSession,
    resize_output_to_input: bool = True,
    device="cuda"
) -> Image.Image:
    import torch
    device = torch.device(device)  # Convert device string to torch.device

    # 1. Preprocess input on GPU
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2. Prepare IO binding
    io_binding = session.io_binding()

    def bind_tensor(name, tensor):
        io_binding.bind_input(
            name=name,
            device_type=device.type,  # e.g., 'cuda'
            device_id=device.index if device.index is not None else 0,
            element_type=np.float32 if tensor.dtype == torch.float32 else np.int64,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    # Bind inputs
    bind_tensor("pixel_values", pixel_values)
    bind_tensor("input_ids", input_ids)
    bind_tensor("attention_mask", attention_mask)

    # 3. Prepare GPU output buffer
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape  # e.g. [1, 1, None, None]
    # Convert dynamic dimensions to fixed integers.
    fixed_output_shape = []
    for i, dim in enumerate(output_shape):
        if isinstance(dim, int):
            fixed_output_shape.append(dim)
        else:
            # For dynamic dimensions, assume they match the corresponding dimension of the input.
            # For CLIPSeg, outputs are typically of shape [1, 1, H, W] where H, W come from the input.
            if i < len(pixel_values.shape):
                fixed_output_shape.append(pixel_values.shape[i])
            else:
                fixed_output_shape.append(224)  # fallback fixed size if needed
    fixed_output_shape = tuple(fixed_output_shape)

    output_array = torch.empty(fixed_output_shape, dtype=torch.float32, device=device)
    io_binding.bind_output(
        name=output_name,
        device_type=device.type,
        device_id=device.index if device.index is not None else 0,
        element_type=np.float32,
        shape=fixed_output_shape,
        buffer_ptr=output_array.data_ptr(),
    )

    # 4. Run inference with I/O binding
    session.run_with_iobinding(io_binding)

    # 5. Copy output to CPU for visualization
    prob_map = output_array.squeeze().cpu().numpy()  # shape: (H, W)

    # 6. Normalize and convert to mask
    min_val = np.min(prob_map)
    max_val = np.max(prob_map)
    if max_val - min_val < 1e-9:
        scaled = np.zeros_like(prob_map, dtype=np.float32)
    else:
        scaled = (prob_map - min_val) / (max_val - min_val)
    mask_pil = Image.fromarray((scaled * 255).astype(np.uint8))

    if resize_output_to_input:
        mask_pil = mask_pil.resize(image.size, resample=Image.BILINEAR)

    return mask_pil


def clipseg_preprocess(frame_bgra):
    # Convert BGRA → BGR (strip alpha), then → RGB
    bgr = frame_bgra[..., :3]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # # Resize + center crop with OpenCV
    # resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    # top = (256 - 224) // 2
    # left = (256 - 224) // 2
    # cropped = resized[top:top+224, left:left+224]

    # # For inference (e.g. CLIPSeg), convert to tensor + normalize
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # norm = ((cropped / 255.0) - mean) / std
    # tensor = norm.transpose(2, 0, 1)  # CHW

    return Image.fromarray(rgb)

def preprocess_vga_image(image: Image.Image, target_size=(224, 224)) -> Image.Image:
    # Center crop to square
    min_dim = min(image.width, image.height)
    left = (image.width - min_dim) // 2
    top = (image.height - min_dim) // 2
    cropped = image.crop((left, top, left + min_dim, top + min_dim))

    # Resize to model input size
    return cropped.resize(target_size, Image.BICUBIC)

def clipseg_postprocess(mask_np: np.ndarray) -> torch.Tensor:
    """
    Mimics Albumentations pipeline for colorized masks:
        Resize(256, 256) → CenterCrop(224, 224) → Normalize → ToTensorV2

    Input: RGB image (H, W, 3), dtype=uint8
    Output: PyTorch tensor (3, 224, 224), dtype=float32
    """
    import torch
    import cv2

    # Step 1: Resize to 256x256
    resized = cv2.resize(mask_np, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Step 2: Center crop to 224x224
    top = (256 - 224) // 2
    left = (256 - 224) // 2
    cropped = resized[top:top+224, left:left+224]

    # Step 3: Normalize (ImageNet-style)
    cropped = cropped.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    normalized = (cropped - mean) / std

    # Step 4: Convert to PyTorch tensor (CHW)
    chw = normalized.transpose(2, 0, 1)
    return torch.from_numpy(chw).float()



def init_clipseg_model(
    model_name: str = "CIDAS/clipseg-rd64-refined",#"CIDAS/clipseg-rd16",
    device: str = None
):
    """
    Load the CLIPSeg model and processor from Hugging Face.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = CLIPSegProcessor.from_pretrained(model_name, use_fast=True)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name)
    # model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.to(device)
    
    return processor, model, device

def clipseg_inference(
    image: Image.Image,
    prompt: str,
    processor: CLIPSegProcessor,
    model: CLIPSegForImageSegmentation,
    device: str,
    resize_output_to_input: bool = True
) -> Image.Image:
    """
    Given a PIL image and a text prompt, return a single-channel mask 
    (0-255) from CLIPSeg representing semantic probability.
    """
    # 1. Preprocess inputs for CLIPSeg
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    
    # 2. Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        # with torch.cuda.amp.autocast():
        #     outputs = model(**inputs)
        # Use outputs.logits instead of outputs.pred_masks (depends on HF version)
        prob_map = outputs.logits #.sigmoid()  # shape: [batch_size, 1, height, width]

    # 3. Extract the single-channel prob map and convert to PIL
    prob_map_2d = prob_map[0].cpu().numpy()

    # (B) Rescale the clipped logits to [0..1]
    current_min = prob_map_2d.min()
    current_max = prob_map_2d.max()
    range_val = current_max - current_min

    if range_val < 1e-9:
        # If the entire map is the same value, everything becomes 0
        scaled_prob_map = np.zeros_like(prob_map_2d, dtype=np.float32)
    else:
        scaled_prob_map = (prob_map_2d - current_min) / range_val  # => [0..1]

    mask_pil = Image.fromarray((scaled_prob_map * 255).astype(np.uint8))
    
    # 4. Optionally resize back to original resolution
    if resize_output_to_input:
        mask_pil = mask_pil.resize(image.size, resample=Image.BILINEAR)
    
    return mask_pil

#########################################
# 2. Optional Plotting of the First Frame
#########################################

def show_frame_and_mask(original: Image.Image, mask: Image.Image, alpha: float = 0.6):
    """
    Display the original RGB frame next to the CLIPSeg probability heatmap
    using Matplotlib.
    """
    # Convert images to NumPy
    frame_np = np.array(original)
    mask_np = np.array(mask) / 255.0  # scale to [0..1] for colormap

    # Set up a side-by-side plot
    plt.figure(figsize=(12, 6))

    # Left: Original image
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(frame_np)
    ax1.set_title("Original Frame")
    ax1.axis("off")

    # Right: Overlay heatmap
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(frame_np, alpha=1.0)
    ax2.imshow(mask_np, cmap="turbo", alpha=alpha)
    ax2.set_title("CLIPSeg Probability Heatmap")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()

def colorize_mask(mask_np, cmap_name="turbo"):
    """
    Convert single-channel mask (H, W) into an RGB image (H, W, 3)
    using a Matplotlib colormap (e.g., "turbo").
    """
    # 1) Convert mask to float in [0..1]
    mask_norm = mask_np.astype(np.float32) / 255.0

    # 2) Get a colormap from Matplotlib
    cmap = matplotlib.colormaps[cmap_name]

    # 3) Apply the colormap -> returns RGBA (H, W, 4)
    rgba = cmap(mask_norm)  # each pixel: [R, G, B, A], each in [0..1]

    # 4) Take only the RGB channels, convert to [0..255] uint8
    rgb = (rgba[..., :3] * 255).astype(np.uint8)  # shape (H, W, 3)

    return rgb

# Precompute the LUT once
def get_colormap_lut(cmap_name="turbo", lut_size=256):
    cmap = plt.get_cmap(cmap_name, lut_size)
    lut = (cmap(np.linspace(0, 1, lut_size))[:, :3] * 255).astype(np.uint8)  # shape: (256, 3)
    return lut  # Each row is an RGB color

# Fast LUT-based colorizer
def colorize_mask_fast(mask_np, lut):
    """
    Convert a (H, W) uint8 mask into a (H, W, 3) RGB image using the provided LUT.
    """
    return lut[mask_np]  # Very fast NumPy indexing: shape (H, W, 3)


#########################
# 3. Main Capture Logic #
#########################

def run_clipseg(
        camera_available: bool = False,
        video_available: str = None,
        onnx_available: bool = False,
        show_first_frame: bool = False,
        export_onnx: bool = False,
    ):
    
    # Set paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    # onnx_path = os.path.join(project_root, "clipseg.onnx")
    onnx_path = os.path.join(os.path.dirname(project_root), "cohorts", "clipseg.onnx")
    test_space_path = os.path.join(os.path.dirname(project_root), "test_space")

    processor, model, device = init_clipseg_model()

    try:
        so = ort.SessionOptions()
        so.log_severity_level = 1
        session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CUDAExecutionProvider"])
        io_binding = session.io_binding()
    except Exception as e:
        print(f"[WARNING] Failed to initialize ONNX session with CUDAExecutionProvider: {e}")

    lut = get_colormap_lut("turbo")

    # Prompt for segmentation
    prompt = "cabinet"  # Example prompt

    # Set up camera, if available
    if camera_available:
        try:
            pipeline = ch.get_camera(360, 640, 30)
            time.sleep(3)  # warm-up
        except:
            print("Pipeline Unsuccessful")
            return

    # Prepare video writer (imageio)
    frames_list = []
    iteration_times = []
    timestamps = []

    duration = 10.0
    start_time = time.time()


    if camera_available:
        try:
            while time.time() - start_time < duration:
                iteration_start = time.time()

                # 1. Grab a frame from your camera (NumPy, shape=(360,640,3)?)
                real_image, timestamp = get_image(pipeline)

                # Downscale the image to 224x224
                # process_image(real_image)

                # real_rgb_image = cv2.cvtColor(real_image, cv2.COLOR_BGRA2RGB)
                
                # 2. Convert to PIL
                # If frames are BGR, convert BGR->RGB with [..., ::-1]
                # frame_pil = Image.fromarray(real_rgb_image)

                # print(f"Real Image shape: {real_rgb_image.shape}")
                # print(f"Frame shape: {frame_pil.size}")
                frame_pil = clipseg_preprocess(real_image)
                

                # 3. Run CLIPSeg
                try:
                    mask_pil = clipseg_inference_onnx_gpu(frame_pil, prompt, processor, session)
                    # mask_pil = clipseg_inference(
                    #     image=frame_pil,
                    #     prompt=prompt,
                    #     processor=processor,
                    #     model=model,
                    #     device=device
                    # )

                    # 4. Convert single-channel PIL -> NumPy for video writing
                    mask_np = np.array(mask_pil)  # shape (360, 640), dtype=uint8, range [0..255]
                    # colorized = colorize_mask(mask_np, cmap_name="turbo")
                    mask_np = colorize_mask_fast(mask_np, lut)
                    # colorized = np.array(frame_pil)  # shape (360, 640, 3), dtype=uint8, range [0..255]

                    # svnet_input = clipseg_postprocess(mask_np)

                    # Measure iteration times for frame averaging
                    iteration_end = time.time()
                    iteration_times.append(iteration_end - iteration_start)
                    # timestamps.append(timestamp)

                    # 5. Write to video
                    frames_list.append(mask_np)

                    # 6. Optionally show the first frame
                    if show_first_frame:
                        show_frame_and_mask(frame_pil, mask_pil, alpha=0.6)
                        show_first_frame = False
                
                except:
                    # Export the model to ONNX format
                    export_onnx_model(onnx_path, 
                                    dummy_image=frame_pil, dummy_text=prompt, 
                                    processor=processor, model=model, device=device
                    )
                    print("Exported ONNX model successfully. Exiting Forcibly")
                    break
        finally:
            close_camera(pipeline)
            if timestamps:
                # Convert timestamps (ms) to seconds and calculate differences
                iteration_durations = np.diff(iteration_times) / 1000.0  # Convert ms to seconds
                avg_time = np.mean(iteration_durations)
                avg_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"Average iteration time: {avg_time:.4f} s -> ~{avg_fps:.2f} FPS")
            elif iteration_times:
                avg_inf_time = sum(iteration_times) / len(iteration_times)
                avg_fps = 1.0 / avg_inf_time
                print(f"Average inference time: {avg_inf_time:.3f}s")
                print(f"Effective inference FPS: {avg_fps:.2f}")
            else:
                avg_fps = 30  # fallback
        
        output_path = os.path.join(project_root, "datafiles", "clipseg_output.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames_list, fps=avg_fps)
        print(f"Video saved as {output_path}")
    else:
        video_path = os.path.join(test_space_path, f"{video_available}.MOV")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        print(f"Video path: {video_path}")

        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            traceback.print_exc()
            return
        
        start_offset = 45.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / fps  # seconds between frames
        last_seek_msec = -1  # initialize outside loop
        seek_threshold_msec = 1000.0 / fps

        # Jump to start_offset in the video
        processing_start_msec = start_offset * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, processing_start_msec)


        start_time = time.time()
        frame_idx = 0

        # Stats for measuring inference performance
        # print(session.get_providers())

        while cap.isOpened():
            now = time.time()
            elapsed_wall_time = now - start_time

            # Stop after desired real-time duration
            if elapsed_wall_time >= duration:
                print(f"Reached duration of {duration:.2f} seconds.")
                break

            # Target time for this frame (in wall-clock time)
            target_time = start_time + frame_idx * frame_interval

            # If we're too early for this frame, wait
            if now < target_time:
                time.sleep(target_time - now)
                continue

            # Compute corresponding time in the video
            current_video_msec = processing_start_msec + elapsed_wall_time * 1000

            # Only seek if we're skipping ahead meaningfully
            if current_video_msec - last_seek_msec >= seek_threshold_msec:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_video_msec)
                last_seek_msec = current_video_msec

            ret, real_image = cap.read()
            if not ret:
                print("End of video reached.")
                break

            # --- TIMING ---
            iteration_start = time.time()

            # --- YOUR IMAGE PROCESSING PIPELINE ---
            # process_image(real_image)  # e.g., resize to 224x224
            # # Convert to RGB
            # real_rgb_image = cv2.cvtColor(real_image, cv2.COLOR_BGRA2RGB)
            # # Convert to PIL
            # frame_pil = Image.fromarray(real_rgb_image)
            frame_pil = clipseg_preprocess(real_image)

            # Run CLIPSeg
            # if onnx_available:
            mask_pil = clipseg_inference_onnx(frame_pil, prompt, processor, session)
            # else:
                # mask_pil = clipseg_inference(frame_pil, prompt, processor, model, device)

            mask_np = np.array(mask_pil)  # shape (360, 640), dtype=uint8, range [0..255]

            # Convert for writing (colorized or original RGB)
            # mask_np = colorize_mask(mask_np, cmap_name="turbo")
            mask_np = colorize_mask_fast(mask_np, lut)

            svnet_input = clipseg_postprocess(mask_np)

            # Save time & output
            iteration_end = time.time()
            iteration_times.append(iteration_end - iteration_start)
            frames_list.append(mask_np)

            if export_onnx:
                # Export the model to ONNX format
                export_onnx_model(onnx_path, 
                                dummy_image=frame_pil, dummy_text=prompt, 
                                processor=processor, model=model, device=device
                )
                break

            frame_idx += 1

        cap.release()

        # Print inference stats
        # print(iteration_times)
        avg_inf_time = sum(iteration_times) / len(iteration_times)
        print(f"Average inference time: {avg_inf_time:.3f}s")
        print(f"Effective inference FPS: {1.0 / avg_inf_time:.2f}")

        output_path = os.path.join(project_root, "clipseg_output.mp4")

        # Compute average FPS based on actual processing time
        effective_fps = len(frames_list) / sum(iteration_times)

        # Optionally convert to uint8 if needed
        frames_list = [frame.astype(np.uint8) for frame in frames_list]

        # Save video
        playback_fps = min(fps, effective_fps)
        imageio.mimsave(output_path, frames_list, fps=playback_fps)
        print(f"Video saved as {output_path} at {playback_fps:.2f} FPS")
#%%
VID_AVAIL = False
CAM_AVAIL = True
ONNX_AVAIL = False
run_clipseg(camera_available=CAM_AVAIL,
            video_available=VID_AVAIL,
            onnx_available=ONNX_AVAIL,
            show_first_frame=False,
            export_onnx=False)
# %%