#%%
import numpy as np
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
import matplotlib.pyplot as plt
import os
import time

import torch
import cv2
from PIL import Image

# import flight.command_helper as ch
import sousvide.flight.zed_command_helper as ch 

from abc import abstractmethod, abstractproperty
from dataclasses import dataclass, field
from typing import Any, Tuple, Type, List
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose
import imageio
from tqdm import tqdm
from einops import rearrange
import gc

import torch.nn.functional as F
import matplotlib.cm as cm

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # Disable Albumentations update check

try:
    import synthesize.nerf_utils as nf
except:
    print("Unable to load NeRF/3DGS utilities. This is expected if you are not using nerfstudio.")

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
except:
    print("Unable to load CLIPSeg. This is expected if you are not using CLIPSeg.")


#%%##########################################
# 1. Initialize CLIPSeg Model and Processor #
#############################################
class CLIPSegONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values, input_ids, attention_mask):
        return self.model(pixel_values=pixel_values,
                          input_ids=input_ids,
                          attention_mask=attention_mask).logits

def init_clipseg_model(
    model_name: str = "CIDAS/clipseg-rd64-refined",
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
# def init_clipseg_model(
#     model_name: str = "CIDAS/clipseg-rd64-refined",
#     device: str = None,
#     export_onnx: bool = False,
#     onnx_path: str = "clipseg.onnx"
# ):
#     """
#     Load the CLIPSeg model and processor from Hugging Face.
#     Optionally export to ONNX format.

#     Returns:
#         processor, model, device
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     processor = CLIPSegProcessor.from_pretrained(model_name,use_fast=True)
#     model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
#     model.eval()

#     if export_onnx:
#         # Dummy input (batch of 1 image and 1 text prompt)
#         dummy_image = Image.new("RGB", (224, 224), color="white")  # typical clipseg size
#         dummy_text = ["snow"]

#         inputs = processor(images=dummy_image, text=dummy_text, return_tensors="pt")
#         # inputs = {k: v.to(device) for k, v in inputs.items()}

#         pixel_values = inputs["pixel_values"]  # shape: [1, 3, H, W]
#         input_ids = inputs["input_ids"]        # shape: [1, seq_len]
#         attention_mask = inputs["attention_mask"]  # shape: [1, seq_len]

#         # Sanity check shapes
#         assert pixel_values.ndim == 4
#         assert input_ids.ndim == 2

#         # Export model
#         torch.onnx.export(
#             model,
#             (inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"]),
#             onnx_path,
#             input_names=["pixel_values", "input_ids", "attention_mask"],
#             output_names=["logits"],
#             dynamic_axes={
#                 "pixel_values": {0: "batch", 2: "height", 3: "width"},
#                 "input_ids": {0: "batch", 1: "sequence"},
#                 "attention_mask": {0: "batch", 1: "sequence"},
#                 "logits": {0: "batch", 1: "height", 2: "width"},
#             },
#             opset_version=13
#         )
#         print(f"ONNX model exported to: {onnx_path}")

#     return processor, model, device

def export_onnx_model(
    onnx_path,
    dummy_image,dummy_text,
    processor,model,device
    ):

    inputs = processor(images=dummy_image, text=dummy_text, return_tensors="pt")
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    pixel_values = inputs["pixel_values"]  # shape: [1, 3, H, W]
    input_ids = inputs["input_ids"]        # shape: [1, seq_len]
    attention_mask = inputs["attention_mask"]  # shape: [1, seq_len]

    # Sanity check shapes
    assert pixel_values.ndim == 4
    assert input_ids.ndim == 2

    pixel_values = pixel_values.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Wrap model to avoid tuple input unpacking error
    wrapped_model = CLIPSegONNXWrapper(model).to(device)
    wrapped_model.eval()

    # Export ONNX
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
        opset_version=13
    )
    print(f"ONNX model exported to: {onnx_path}")
    


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


#########################
# 3. Main Capture Logic #
#########################

def run_clipseg():
    project_root = os.path.dirname(os.path.abspath(__file__))
    # mcts_list_path = os.path.join(project_root, "mcts_trees.pkl")
    # mcts_txt_path = os.path.join(project_root, "formatted_trees.txt")
    onnx_path = os.path.join(project_root, "clipseg.onnx")

    # A. Initialize CLIPSeg
    processor, model, device = init_clipseg_model()

    transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])

    process_image = lambda x: transform(image=x)["image"]

    # B. Prompt for segmentation
    prompt = "person"  # Example prompt

    # C. Set up camera
    try:
        pipeline = ch.get_camera(360, 640, 30)
        time.sleep(3)  # warm-up
    except:
        print("Pipeline Unsuccessful")
        return

    # D. Prepare video writer (imageio)
    frames_list = []
    iteration_times = []

    duration = 10.0
    start_time = time.time()

    # Show the first frame only
    show_first_frame = False

    # Export model to ONNX file
    export_onnx = False

    try:
        while time.time() - start_time < duration:
            iteration_start = time.time()

            # 1. Grab a frame from your camera (NumPy, shape=(360,640,3)?)
            real_image, timestamp = ch.get_image(pipeline)

            # Downscale the image to 224x224
            process_image(real_image)

            real_rgb_image = cv2.cvtColor(real_image, cv2.COLOR_BGRA2RGB)
            
            # 2. Convert to PIL
            # If frames are BGR, convert BGR->RGB with [..., ::-1]
            frame_pil = Image.fromarray(real_rgb_image)

            # print(f"Real Image shape: {real_rgb_image.shape}")
            # print(f"Frame shape: {frame_pil.size}")

            # 3. Run CLIPSeg
            mask_pil = clipseg_inference(frame_pil, prompt, processor, model, device)

            # 4. Convert single-channel PIL -> NumPy for video writing
            # mask_np = np.array(mask_pil)  # shape (360, 640), dtype=uint8, range [0..255]
            # colorized = colorize_mask(mask_np, cmap_name="turbo")
            colorized = np.array(frame_pil)  # shape (360, 640, 3), dtype=uint8, range [0..255]

            # Measure iteration times for frame averaging
            # iteration_end = time.time()
            # iteration_duration = iteration_end - iteration_start
            iteration_times.append(timestamp)

            # 5. Write to video
            frames_list.append(colorized)
            # frames_list.append(mask_np)

            # 6. Optionally show the first frame
            if show_first_frame:
                show_frame_and_mask(frame_pil, mask_pil, alpha=0.6)
                show_first_frame = False
            
            if export_onnx:
                # Export the model to ONNX format
                export_onnx_model(onnx_path, 
                                  dummy_image=frame_pil, dummy_text=prompt, 
                                  processor=processor, model=model, device=device
                )
                break

    finally:
        ch.close_camera(pipeline)

        if iteration_times:
            # Convert timestamps (ms) to seconds and calculate differences
            iteration_durations = np.diff(iteration_times) / 1000.0  # Convert ms to seconds
            avg_time = np.mean(iteration_durations)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Average iteration time: {avg_time:.4f} s -> ~{avg_fps:.2f} FPS")
        else:
            avg_fps = 30  # fallback
        
        output_path = os.path.join(project_root, "clipseg_output.mp4")
        imageio.mimsave(output_path, frames_list, fps=avg_fps)
        print("Video saved as clipseg_output.mp4")

run_clipseg()