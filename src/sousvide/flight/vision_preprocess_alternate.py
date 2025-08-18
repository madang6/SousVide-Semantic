import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage.segmentation import slic, find_boundaries
from skimage.metrics import structural_similarity as ssim
from scipy.stats import mode

import torch
import torch.nn.functional as F
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")
import onnxruntime as ort

import cv2
import imageio
from PIL import Image
import albumentations as A

from typing import Any, List, Optional, Tuple, Type, Union

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class CLIPSegHFModel:
    def __init__(
        self,
        hf_model: str = "CIDAS/clipseg-rd64-refined",
        device: Optional[str] = None,
        cmap: str = "turbo",
        onnx_model_path: Optional[str] = None,
        onnx_model_fp16_path: Optional[str] = None
    ):
        """
        HuggingFace CLIPSeg wrapper for torch inference, with optional ONNX fallback.
        If you pass onnx_model_path and it doesn’t exist, we’ll export it automatically.
        """
        # select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # load HF processor
        self.processor = CLIPSegProcessor.from_pretrained(hf_model, use_fast=True)

        # color‐lut, running bounds, caches, etc.
        self.lut = get_colormap_lut(cmap_name=cmap)
        self.running_min = float('inf')
        self.running_max = float('-inf')
        self.prev_image = None
        self.prev_output = None
        self.segmentation_ema = None
        self.ema_alpha = 0.7
        self.last_superpixel_mask = None
        self.superpixel_every = 1
        self.frame_counter = 0

        # Baseline for calibration
        self.loiter_max = 0.0
        self.loiter_area_frac = 0.0
    #FIXME
        self.loiter_mask = None            # uint8 mask of best region from calibration
        self.loiter_contour = None         # optional: for drawing crisp outlines
        self.iou_thresh = 0.50             # tune 0.4–0.6
        self.overlay_alpha = 0.40
        self.overlay_color = (0, 255, 0)   # RGB

        self.loiter_cnt = None
        self.loiter_solidity = None
        self.loiter_eccentricity = None
        self.shape_thresh = 0.375           # tune 0.05–0.20 (lower = stricter)
        self.area_tolerance = 0.15         # ±15%
        self.sol_tol = 0.10                # ±10% band
        self.ecc_tol = 0.15                # ±15% band
    #
        # ONNX support
        self.use_onnx = False
        self.ort_session = None
        self.using_fp16 = False
        if onnx_model_path is not None:
            if ort is None:
                raise ImportError(
                    "onnxruntime is not installed; pip install onnxruntime to use ONNX inference."
                )
            # if the .onnx file is missing, export it:
            if not os.path.isfile(onnx_model_path):
                print(f"Exporting HF model to ONNX at {onnx_model_path}...")
                self.model = CLIPSegForImageSegmentation.from_pretrained(hf_model)
                self.model.to(self.device).eval()
                self._export_onnx(onnx_model_path)
            if onnx_model_fp16_path:
                print(f"Converting ONNX model to FP16 at {onnx_model_fp16_path}...")
                if (not os.path.isfile(onnx_model_fp16_path)):
                    self._convert_to_fp16(onnx_model_path, onnx_model_fp16_path)
                onnx_model_path = onnx_model_fp16_path
                self.using_fp16 = True
            print(f"Using ONNX model at {onnx_model_path}...")
            
            so = ort.SessionOptions()

            # so.enable_mem_pattern     = False
            # so.enable_cpu_mem_arena   = False
            
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            
            # so.intra_op_num_threads = 1
            # so.inter_op_num_threads = 1
            so.log_severity_level   = 2
            self.ort_session = ort.InferenceSession(
                onnx_model_path,
                sess_options=so,
                providers=["CUDAExecutionProvider"]#, "CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.io_binding = self.ort_session.io_binding()
            self.use_onnx = True

            self._input_gpu  = None
            self._output_gpu = None

        else:
            self.model = CLIPSegForImageSegmentation.from_pretrained(hf_model)
            self.model.to(self.device).eval()

    def _export_onnx(self, onnx_path: str):
        """
        Exports the HF CLIPSeg model to ONNX at `onnx_path` using a model wrapper,
        grabs a real frame from the Zed camera at the configured resolution/fps.
        """
        import sousvide.flight.zed_command_helper as zed

        # 1) Grab one frame from the Zed camera
        camera = zed.get_camera(height=376, width=672, fps=30)
        if camera is None:
            raise RuntimeError("Unable to initialize Zed camera.")
        frame, timestamp = zed.get_image(camera)
        if not isinstance(frame, np.ndarray):
            raise RuntimeError(f"Expected NumPy array from Zed, got {type(frame)}")
        # Convert BGRA/BGR → RGB PIL.Image
        bgr = frame[..., :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 2) Prepare dummy text & tokenize image+text
        dummy_prompt = "a cabinet"
        proc_inputs = self.processor(
            images=pil_img,
            text=dummy_prompt,
            return_tensors="pt"
        )
        proc_inputs = {k: v.to(self.device) for k, v in proc_inputs.items()}
        inputs = (
            proc_inputs["input_ids"],
            proc_inputs["pixel_values"],
            proc_inputs["attention_mask"],
        )

        # 3) Wrap HF model so it outputs a single tensor (.logits)
        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, pixel_values, attention_mask):
                out = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask
                )
                return out.logits

        wrapper = Wrapper(self.model).eval().to(self.device)

        # 4) Export with generic names, dynamic batch+seq_len axes, opset 21
        from onnx import __version__, IR_VERSION
        from onnx.defs import onnx_opset_version
        print(f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")
        print(f"[INFO] Exporting ONNX model to '{onnx_path}'…")
        torch.onnx.export(
            wrapper,
            inputs,
            onnx_path,
            input_names=["input_ids", "pixel_values", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq_len"},
                "pixel_values":   {0: "batch"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "logits":         {0: "batch", 2: "height", 3: "width"}
            },
            opset_version=20,
            do_constant_folding=True,
            verbose=False,
        )
        # ── Patch Resize nodes (cubic→linear) ──
        # ONNXRuntime’s Resize only supports 'cubic' on 2-D or certain 4-D inputs,
        # so we rewrite anything using cubic to linear for full compatibility.
        import onnx
        model_onnx = onnx.load(onnx_path)
        patched = False
        for node in model_onnx.graph.node:
            if node.op_type == "Resize":
                for attr in node.attribute:
                    if attr.name == "mode" and attr.s.decode("utf-8").lower() == "cubic":
                        attr.s = b"linear"
                        patched = True
        if patched:
            onnx.save(model_onnx, onnx_path)
            print(f"[INFO] Patched Resize mode→'linear' in {onnx_path}")
        print(f"[INFO] Exported ONNX model with inputs ['input_ids','pixel_values','attention_mask'] → '{onnx_path}' ✅")
    # def _export_onnx(self, onnx_path: str):
    #     """
    #     Exports the HF CLIPSeg model to ONNX at onnx_path.
    #     Uses a dummy text+image input from the processor.
    #     """
    #     # pick a dummy prompt and image
    #     dummy_prompt = "a photo of a cat"
    #     dummy_img = Image.new("RGB", (224, 224), color="white")

    #     # prepare torch inputs
    #     torch_inputs = self.processor(
    #         images=dummy_img,
    #         text=dummy_prompt,
    #         return_tensors="pt"
    #     )
    #     torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}

    #     # names must match the processor/model
    #     # input_names = ["pixel_values", "input_ids", "attention_mask"]
    #     # output_names = ["logits"]
    #     # dynamic_axes = {
    #     #     "pixel_values":   {0: "batch", 2: "height", 3: "width"},
    #     #     "input_ids":      {0: "batch", 1: "seq_len"},
    #     #     "attention_mask": {0: "batch", 1: "seq_len"},
    #     #     "logits":         {0: "batch", 2: "height", 3: "width"},
    #     # }

    #     # export
    #     torch.onnx.export(
    #         self.model,
    #         (
    #             torch_inputs["input_ids"],
    #             torch_inputs["pixel_values"],
    #             torch_inputs["attention_mask"],
    #         ),
    #         onnx_path,
    #         input_names=["input_ids", "pixel_values", "attention_mask"],
    #         # input_names=["input_ids", "pixel_values"],session
    #         output_names=["logits"],
    #         dynamic_axes={
    #             "input_ids":      {1: "seq_len"},
    #             # "pixel_values":   {2: "height", 3: "width"},
    #             "attention_mask": {1: "seq_len"},
    #             # "logits":         {2: "height", 3: "width"},
    #         },
    #         opset_version=17,
    #         do_constant_folding=True,
    #     )
    
    # def _convert_to_fp16(self, onnx_path: str, fp16_path: str):
    #     import onnx
    #     from onnxconverter_common import float16

    #     model = onnx.load("clipseg_model.onnx")

    #     model_fp16 = float16.convert_float_to_float16(
    #         model,
    #         keep_io_types=False,       # keep inputs/outputs in float32
    #         disable_shape_infer=True,  # skip ONNX shape inference
    #         op_block_list=[],
    #         check_fp16_ready=False
    #     )

    #     onnx.save_model(model_fp16, "clipseg_model_fp16.onnx")

    def _convert_to_fp16(self, onnx_path: str, fp16_path: str):
        import onnx
        from onnxconverter_common import float16
        """Convert an ONNX model to FP16 using the standard converter."""

        model = onnx.load(onnx_path)

        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
        )

        onnx.save(model_fp16, fp16_path)

    def _rescale_global(self, arr: np.ndarray) -> np.ndarray:
        """
        Rescale `arr` to [0,1] using the min/max seen so far (updated here).
        """
        cur_min = float(arr.min())
        cur_max = float(arr.max())
        self.running_min = min(self.running_min, cur_min)
        self.running_max = max(self.running_max, cur_max)
        # print(f"Running bounds: min={self.running_min}, max={self.running_max}")
        # print(f"Current bounds: min={cur_min}, max={cur_max}")
        span = self.running_max - self.running_min
        if span < 1e-9:
            scaled = np.zeros_like(arr, dtype=np.float32)
        else:
            scaled = (arr - self.running_min) / span
        return scaled
    
    # def _run_onnx_model(self, img: Image.Image, prompt: str) -> np.ndarray:
    #     """
    #     Runs forward pass via onnxruntime and returns the raw logits as a float32 numpy array.
    #     """
    #     # 1) Get PyTorch tensors from the HF processor...
    #     torch_inputs = self.processor(images=img, text=prompt, return_tensors="pt")
    #     # 2) Move them to CPU & convert to numpy for ONNX runtime
    #     ort_inputs = {}
    #     for inp in self.ort_session.get_inputs():
    #         name = inp.name
    #         tensor = torch_inputs.get(name)
    #         if tensor is None:
    #             continue
    #         # detach, move to CPU, numpy
    #         ort_inputs[name] = tensor.cpu().numpy()
    #     # 3) Run the ONNX session
    #     ort_outs = self.ort_session.run(None, ort_inputs)
    #     # assume first output is logits [1,1,H,W]
    #     logits = ort_outs[0]
    #     return logits.squeeze().astype(np.float32)

    def _run_onnx_model(self, img: Image.Image, prompt: str) -> np.ndarray:
        import numpy as np
        import torch

        # 1) Preprocess — half if FP16, full-precision otherwise
        torch_inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        if self.using_fp16:
            torch_inputs = {
                k: (v.half().to(self.device) if k == "pixel_values"
                    else v.to(self.device))
                for k, v in torch_inputs.items()
            }
        else:
            torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}

        # 2) New binding for each run
        io_bind = self.ort_session.io_binding()

        # 3) Bind *all* ONNX inputs by name
        for meta in self.ort_session.get_inputs():
            name = meta.name
            if name not in torch_inputs:
                raise RuntimeError(f"ONNX model expects '{name}' but no tensor was found.")
            tensor = torch_inputs[name]
            elem_type = (
                np.float16 if tensor.dtype == torch.float16 else
                np.float32 if tensor.dtype == torch.float32 else
                np.int64
            )
            io_bind.bind_input(
                name=name,
                device_type=self.device,   # e.g. "cuda"
                device_id=0,
                element_type=elem_type,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        # 4) Figure out the output shape
        out_meta = self.ort_session.get_outputs()[0]
        B, _, H, W = torch_inputs["pixel_values"].shape
        if len(out_meta.shape) == 3:
            out_shape = (B, H, W)
        elif len(out_meta.shape) == 4:
            C = out_meta.shape[1] if isinstance(out_meta.shape[1], int) else 1
            out_shape = (B, C, H, W)
        else:
            raise RuntimeError(f"Unsupported logits rank: {len(out_meta.shape)}")

        # 5) Allocate & bind output on GPU
        out_dtype = torch.float16 if self.using_fp16 else torch.float32
        gpu_out = torch.empty(out_shape, dtype=out_dtype, device=self.device)
        io_bind.bind_output(
            name=out_meta.name,
            device_type=self.device,
            device_id=0,
            element_type=(np.float16 if self.using_fp16 else np.float32),
            shape=out_shape,
            buffer_ptr=gpu_out.data_ptr(),
        )

        # 6) Run
        self.ort_session.run_with_iobinding(io_bind)

        # 7) Fetch & postprocess
        result = gpu_out.cpu().numpy()
        # if [B,1,H,W], drop channel
        if result.ndim == 4 and result.shape[1] == 1:
            result = result[:, 0]
        return result.squeeze()
    # def _run_onnx_model(self, img: Image.Image, prompt: str) -> np.ndarray:
    #     import numpy as np
    #     import torch

    #     # 1) Preprocess on GPU
    #     torch_inputs = self.processor(images=img, text=prompt, return_tensors="pt")
    #     if self.using_fp16:
    #         # convert to FP16 if needed and move to device
    #         torch_inputs = {
    #             k: (v.half().to(self.device) if k == "pixel_values" else v.to(self.device))
    #             for k, v in torch_inputs.items()
    #         }
    #     else:
    #         torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}

    #     # 2) Fresh IOBinding
    #     io_binding = self.ort_session.io_binding()

    #     # 3) Bind inputs zero-copy
    #     sess_input_names = {inp.name for inp in self.ort_session.get_inputs()}
    #     for name, tensor in torch_inputs.items():
    #         if name not in sess_input_names:
    #             continue
    #         elem_type = np.float32 if tensor.dtype == torch.float32 else np.int64
    #         io_binding.bind_input(
    #             name=name,
    #             device_type=self.device,   # e.g. "cuda"
    #             device_id=0,
    #             element_type=elem_type,
    #             shape=tuple(tensor.shape),
    #             buffer_ptr=tensor.data_ptr(),
    #         )

    #     # 4) Figure out the ONNX output shape
    #     out_meta = self.ort_session.get_outputs()[0]
    #     B, _, H, W = torch_inputs["pixel_values"].shape
    #     if len(out_meta.shape) == 3:
    #         # [batch, height, width]
    #         out_shape = (B, H, W)
    #     elif len(out_meta.shape) == 4:
    #         # [batch, channels, height, width]
    #         C = out_meta.shape[1] if isinstance(out_meta.shape[1], int) else 1
    #         out_shape = (B, C, H, W)
    #     else:
    #         raise RuntimeError(f"Unsupported logits rank: {len(out_meta.shape)}")

    #     # 5) Allocate & bind output buffer on GPU
    #     out_dtype = torch.float16 if self.using_fp16 else torch.float32
    #     output_gpu = torch.empty(out_shape, dtype=out_dtype, device=self.device)
    #     io_binding.bind_output(
    #         name=out_meta.name,
    #         device_type=self.device,
    #         device_id=0,
    #         element_type=(np.float16 if self.using_fp16 else np.float32),
    #         shape=out_shape,
    #         buffer_ptr=output_gpu.data_ptr(),
    #     )

    #     # 6) Run
    #     self.ort_session.run_with_iobinding(io_binding)

    #     # 7) Fetch & squeeze
    #     result = output_gpu.cpu().numpy()
    #     # if it’s [B, 1, H, W], drop the channel axis
    #     if result.ndim == 4 and result.shape[1] == 1:
    #         result = result[:, 0]
    #     # if batch‐size is 1, you can also drop that:
    #     return result.squeeze()

    def clipseg_hf_inference(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        resize_output_to_input: bool = True,
        use_refinement: bool = False,
        use_smoothing: bool = False,
        scene_change_threshold: float = 1.00,
        verbose=False
    ) -> np.ndarray:
        """
        Run CLIPSeg on a PIL image or numpy array, return colorized mask as (H,W,3) uint8.
        """
        def log(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        # --- Step 1: Convert input to PIL + NumPy ---
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
            image_np = image
        elif isinstance(image, Image.Image):
            img = image
            image_np = np.array(image)
        else:
            raise TypeError(f"Unsupported image type {type(image)}")

        # --- Step 2: Determine whether to reuse ---
        should_reuse = False
        ssim_score = None
        if scene_change_threshold < 1.0 and self.prev_image is not None and self.prev_output is not None:
            # Compare resized grayscale SSIM
            prev_small = cv2.resize(self.prev_image, (64, 64))
            curr_small = cv2.resize(image_np, (64, 64))
            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            ssim_score = ssim(prev_gray, curr_gray, data_range=1.0)
            should_reuse = ssim_score >= scene_change_threshold
            log(f"[DEBUG] SSIM = {ssim_score:.4f}, Threshold = {scene_change_threshold}, Reuse = {should_reuse}")

        # --- Step 3: Reuse path (warp previous mask) ---
        if should_reuse:
            mask_u8 = warp_mask(self.prev_image, image_np, self.prev_output)
            # Optional: fast filtering
            mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=75, sigmaSpace=75)
            colorized = colorize_mask_fast(mask_u8, self.lut)
            overlayed = blend_overlay_gpu(image_np, colorized)
            return overlayed

        # --- Step 4: Run inference (scene has changed) ---
        start = time.time()
        if self.use_onnx:
            arr = self._run_onnx_model(img, prompt)
        else:
            torch_inputs = self.processor(images=img, text=prompt, return_tensors="pt")
            torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}
            with torch.no_grad():
                logits = self.model(**torch_inputs).logits
            arr = logits.cpu().squeeze().numpy().astype(np.float32)

        skip_norm = prompt.strip().lower() == "null"
        if skip_norm:
            prob = 1.0 / (1.0 + np.exp(-arr))
            mask_u8 = (prob * 255).astype(np.uint8)
        else:
#FIXME
            prob = 1.0 / (1.0 + np.exp(-arr))
            # regular_prob = (prob * 255).astype(np.uint8)
            cur_prob_max = float(arr.max())
            self._max_prob_logit = max(getattr(self, "_max_prob_logit", cur_prob_max), cur_prob_max)
            global_max_prob = 1.0 / (1.0 + np.exp(-self._max_prob_logit))
           # —– 4) scale your current mask so that
           #      prob == global_max_prob → 1.0 (full brightness),
           #      lower probs → proportionally dimmer
            prob_scaled = prob / (global_max_prob + 1e-8)
            prob_scaled = np.clip(prob_scaled, 0.0, 1.0)
#
            scaled = self._rescale_global(arr)
            mask_u8 = (scaled * 255).astype(np.uint8)

        if resize_output_to_input:
#FIXME
            # prob_scaled = np.array(Image.fromarray(prob).resize(img.size, resample=Image.BILINEAR))
            prob_scaled = np.array(Image.fromarray(prob_scaled).resize(img.size, resample=Image.BILINEAR))
#
            mask_u8 = np.array(Image.fromarray(mask_u8).resize(img.size, resample=Image.BILINEAR))

        # --- Step 5: Post-processing only on fresh inference ---
        if use_smoothing:
            # Temporal EMA
            mask_f = mask_u8.astype(np.float32)
            if self.segmentation_ema is None or self.segmentation_ema.shape != mask_f.shape:
                self.segmentation_ema = mask_f.copy()
            else:
                self.segmentation_ema = (
                    self.ema_alpha * mask_f + (1 - self.ema_alpha) * self.segmentation_ema
                )
            mask_u8 = np.clip(self.segmentation_ema, 0, 255).astype(np.uint8)

            # Optional filtering
            mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=75, sigmaSpace=75)

        # Optional: superpixel refinement
        if use_refinement:
            self.frame_counter += 1
            if self.frame_counter % self.superpixel_every == 0:
                # self.last_superpixel_mask = superpixel_smoothing(image_np, mask_u8)
                self.last_superpixel_mask = fast_superpixel_seeds(image_np, mask_u8)
            if self.last_superpixel_mask is not None:
                mask_u8 = self.last_superpixel_mask

        # --- Step 6: Render and cache ---
        colorized = colorize_mask_fast(mask_u8, self.lut)
#FIXME
        # regular_prob = colorize_mask_fast(regular_prob, self.lut)
#
        overlayed = blend_overlay_gpu(image_np, colorized)

        # Store just raw mask + image for reuse
        self.prev_image = image_np.copy()
        self.prev_output = mask_u8.copy()

        end = time.time()
        log(f"CLIPSeg inference time: {end - start:.3f} seconds")
        return overlayed, prob_scaled
    
#FIXME
    def _largest_contour_from_mask(self, mask_u8: np.ndarray):
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0, 0.0, 0.0
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else area
        solidity = (area / hull_area) if hull_area > 0 else 1.0
        # Eccentricity via PCA on contour points
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) >= 5:
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
            l1, l2 = float(eigenvalues[0][0]), float(eigenvalues[1][0]) if eigenvalues.shape[0] > 1 else (1.0, 1.0)
            eccentricity = (1.0 - (min(l1,l2) / max(l1,l2))) if max(l1,l2) > 0 else 0.0
        else:
            eccentricity = 0.0
        return cnt, area, solidity, eccentricity

    def _match_shape_distance(self, cnt_ref, cnt_cur) -> float:
        # I1 is a good default; I2/I3 are alternatives
        return cv2.matchShapes(cnt_ref, cnt_cur, cv2.CONTOURS_MATCH_I1, 0.0)
    
    def _area_targeted_mask(self, logits: np.ndarray, target_frac: float,
                            ksize: int = 3, do_open_close: bool = True) -> np.ndarray:
        """
        Build a binary mask whose pixel fraction ≈ target_frac by thresholding at the
        corresponding quantile. Then clean it with light morphology.
        Returns 0/255 uint8.
        """
        H, W = logits.shape
        total = H * W
        target_frac = float(np.clip(target_frac, 1e-4, 0.90))  # safety
        q = 1.0 - target_frac
        # Quantile threshold (clip to sane min/max so we don't get extremes)
        t = np.quantile(logits, q)
        mask = (logits >= t).astype(np.uint8) * 255

        if do_open_close:
            kernel = np.ones((ksize, ksize), np.uint8)
            # close gaps then open speckles
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask
    
    def _make_overlay(self, frame_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        overlay = frame_bgr.copy()
        color_img = np.zeros_like(frame_bgr)
        color_img[mask_u8 > 0] = self.overlay_color
        return cv2.addWeighted(color_img, self.overlay_alpha, overlay, 1.0, 0.0)

    def _compute_iou(self, a_u8: np.ndarray, b_u8: np.ndarray) -> float:
        # Assumes same shape, values are 0/255
        inter = np.count_nonzero((a_u8 > 0) & (b_u8 > 0))
        union = np.count_nonzero((a_u8 > 0) | (b_u8 > 0))
        return float(inter) / float(union) if union else 0.0
    
    def _dbg(self, **k):
        print("[LOITER DBG]", " ".join(f"{kk}={vv}" for kk,vv in k.items()))
#

    def loiter_calibrate(
        self,
        logits: np.ndarray,
        frame_img: np.ndarray,
        active_arm: bool = False
    ) -> Tuple[bool, float, float, Optional[np.ndarray]]:
        """
        Returns (found, sim_score, area_frac, overlay_bgr_or_None)
        - found: whether we matched the stored region (shape-based when active_arm=True)
        - sim_score: global max in logits (unchanged)
        - area_frac: area fraction of the current best region
        - overlay: original frame with current region and/or stored region overlaid
        """
        found = False
        H, W = logits.shape
        total_area = H * W
        sim_score = float(logits.max())

        # 1) threshold → binary mask (per-frame)
        thresh = np.percentile(logits, 90.0)
        mask = (logits >= thresh).astype(np.uint8)  # 0/1

        # 2) connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return found, sim_score, 0.0, None

        # 3) choose the "best" region in this frame:
        best_lab = -1
        best_region_max = -1.0
        best_area = -1
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            region_max = float(logits[labels == lab].max())
            if (region_max > best_region_max) or (region_max == best_region_max and area > best_area):
                best_region_max = region_max
                best_area = area
                best_lab = lab

        # 4) materialize current best region mask (0/255)
        # curr_region_mask = ((labels == best_lab).astype(np.uint8) * 255)

        area_frac = float(best_area) / float(total_area)
        # t0 = time.perf_counter()
        curr_region_mask = self._area_targeted_mask(logits, target_frac=area_frac)
        # t1 = time.perf_counter()
        # logits_rgb = (logits * 255).astype(np.uint8)
        # logits_rgb = colorize_mask_fast(logits_rgb, self.lut)
        # logits_rgb = depth_display_to_rgb(logits, target_hw=(H, W))
        # overlay = self._make_overlay(logits_rgb, curr_region_mask)
        if not active_arm:
            self.overlay_color = (0, 255, 0)
        else:
            self.overlay_color = (0, 0, 255)
        overlay = self._make_overlay(frame_img, curr_region_mask)

        if not active_arm:
            # --- CALIBRATION PHASE ---
            cnt, area_px, solidity, ecc = self._largest_contour_from_mask(curr_region_mask)
            # t2 = time.perf_counter()
            # Should this frame's best become the global reference?
            better = (best_region_max > self.loiter_max) or (
                np.isclose(best_region_max, self.loiter_max, rtol=0, atol=1e-6)
                and (best_area > self.loiter_area_frac * total_area)
            )

            if better and cnt is not None:
                self.loiter_max          = best_region_max
                self.loiter_area_frac    = area_frac
                self.loiter_mask         = curr_region_mask.copy()
                self.loiter_cnt          = cnt
                self.loiter_solidity     = solidity
                self.loiter_eccentricity = ecc

                # Optional visualization of stored outline
                cv2.drawContours(overlay, [self.loiter_cnt], -1, (0, 200, 255), 5)
                # logits_overlay = depth_display_to_rgb(logits, target_hw=(H, W))
                # cv2.drawContours(logits_overlay, [self.loiter_cnt], -1, (0, 200, 255), 2)
                # overlay = logits_overlay

            return found, sim_score, area_frac, overlay

        # --- ACTIVE / ARM PHASE ---
        if self.loiter_cnt is None:
            return found, sim_score, area_frac, overlay

        # rebuild mask to match reference area
        curr_region_mask = self._area_targeted_mask(logits, target_frac=self.loiter_area_frac)

        # recompute current area fraction from the mask you will compare
        cur_area_frac = np.count_nonzero(curr_region_mask) / float(total_area)

        cur_cnt, cur_area_px, cur_sol, cur_ecc = self._largest_contour_from_mask(curr_region_mask)
        if cur_cnt is None:
            return found, sim_score, cur_area_frac, overlay

        # 1) Area sanity using cur_area_frac
        area_ok = abs(cur_area_frac - self.loiter_area_frac) <= self.area_tolerance * self.loiter_area_frac

        # 2) Shape distance (lower is more similar)
        d = self._match_shape_distance(self.loiter_cnt, cur_cnt)
        shape_ok = (d <= self.shape_thresh)

        # 3) Optional morphology bands
        sol_ok = (abs(cur_sol - self.loiter_solidity) <= self.sol_tol * max(self.loiter_solidity, 1e-6))
        ecc_ok = (abs(cur_ecc - self.loiter_eccentricity) <= self.ecc_tol * max(self.loiter_eccentricity, 1e-6))

        print(
            f"[LOITER DBG] d={d:.3f} (thr={self.shape_thresh}) shape_ok={shape_ok} | "
            f"area={cur_area_frac:.3f}, ref={self.loiter_area_frac:.3f}, ok={area_ok} | "
            f"sol={cur_sol:.3f}, ref={self.loiter_solidity:.3f}, ok={sol_ok} | "
            f"ecc={cur_ecc:.3f}, ref={self.loiter_eccentricity:.3f}, ok={ecc_ok}"
        )
        if shape_ok and area_ok and sol_ok and ecc_ok:
            found = True
            # visualize both shapes (current in green, reference outline in orange)
            rgb_overlay = frame_img.copy()
            fill = np.zeros_like(frame_img); fill[curr_region_mask > 0] = (0, 255, 0)
            rgb_overlay = cv2.addWeighted(fill, 0.4, rgb_overlay, 1.0, 0.0)
            cv2.drawContours(rgb_overlay, [self.loiter_cnt], -1, (0, 165, 255), 2)
            cv2.drawContours(rgb_overlay, [cur_cnt], -1, (0, 255, 0), 2)

            logits_overlay = depth_display_to_rgb(logits, target_hw=(H, W))
            fill = np.zeros_like(logits_overlay); fill[curr_region_mask > 0] = (0, 255, 0)
            logits_overlay = cv2.addWeighted(fill, 0.4, logits_overlay, 1.0, 0.0)
            cv2.drawContours(logits_overlay, [self.loiter_cnt], -1, (0, 165, 255), 2)
            cv2.drawContours(logits_overlay, [cur_cnt], -1, (0, 255, 0), 2)

            overlay = rgb_overlay #np.hstack((rgb_overlay, logits_overlay))

        return found, sim_score, area_frac, overlay
                 
################################################
# 2. Lookup Table for Semantic Probability Map #
################################################

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

################################################
# 3. Utility Functions for Image Processing #
################################################
def depth_display_to_rgb(depth_disp: np.ndarray, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """
    Convert ZED VIEW.DEPTH display image to RGB for saving.
    VIEW.DEPTH is 8-bit display; SDK may deliver 1c (GRAY) or 3/4c (BGR/BGRA) depending on settings.
    """
    dv = depth_disp
    # Ensure 8-bit
    if dv.dtype != np.uint8:
        dv = dv.astype(np.uint8, copy=False)

    # Channel handling
    if dv.ndim == 2 or (dv.ndim == 3 and dv.shape[2] == 1):
        rgb = cv2.cvtColor(dv, cv2.COLOR_GRAY2RGB)
    elif dv.ndim == 3 and dv.shape[2] == 3:
        rgb = cv2.cvtColor(dv, cv2.COLOR_BGR2RGB)
    elif dv.ndim == 3 and dv.shape[2] == 4:
        rgb = cv2.cvtColor(dv, cv2.COLOR_BGRA2RGB)
    else:
        # Fallback: replicate to 3 channels
        rgb = np.repeat(dv[..., :1], 3, axis=2)

    if target_hw is not None and rgb.shape[:2] != target_hw:
        H, W = target_hw
        interp = cv2.INTER_AREA if (rgb.shape[1] > W or rgb.shape[0] > H) else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (W, H), interpolation=interp)

    return rgb

def blend_overlay_gpu(base: np.ndarray,
                      overlay: np.ndarray,
                      alpha: float = 1.00) -> np.ndarray:
    """
    Convert `base`→mono-gray on GPU (if needed), resize `overlay` on GPU,
    then blend:  result = α·overlay + (1−α)·gray_base.  Entirely on CUDA.

    Parameters:
        base (np.ndarray): 
            • If 3-channel: H×W×3 BGR/uint8. 
            • If single-channel: H×W/uint8. 
        overlay (np.ndarray): H'×W'×3 BGR/uint8 image to overlay.
        alpha (float): opacity of overlay in [0,1].

    Returns:
        np.ndarray: H×W×3 BGR/uint8 blended result (on CPU).
    """
    # 1. Upload base to GPU and convert to float
    if base.ndim == 2:
        # already gray
        base_gray = torch.from_numpy(base).float().cuda()             # shape [H, W]
    elif base.ndim == 3 and base.shape[2] == 3:
        B = torch.from_numpy(base[:, :, 0]).float().cuda()            # BGR channels
        G = torch.from_numpy(base[:, :, 1]).float().cuda()
        R = torch.from_numpy(base[:, :, 2]).float().cuda()
        # Standard Rec. 601 luma-weights for BGR→gray:
        base_gray = 0.114 * B + 0.587 * G + 0.299 * R               # shape [H, W]
    else:
        raise ValueError(
            "`base` must be H×W (gray) or H×W×3 (BGR).")

    H, W = base_gray.shape

    # 2. Upload overlay to GPU as a float tensor [3, H', W']
    if overlay.ndim != 3 or overlay.shape[2] != 3:
        raise ValueError("`overlay` must be H'×W'×3 (BGR/uint8).")
    ov = torch.from_numpy(overlay).float().permute(2, 0, 1).cuda()    # [3, H', W']
    ov = ov.unsqueeze(0)                                             # [1, 3, H', W']

    # 3. Resize overlay to match base’s H×W (bilinear on GPU)
    ov_resized = F.interpolate(
        ov, size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(0)                                                     # [3, H, W]

    # 4. Stack gray→3 channels: [3, H, W]
    gray3 = base_gray.unsqueeze(0).repeat(3, 1, 1)                    # [3, H, W]

    # 5. Blend on GPU: α·overlay + (1−α)·gray3
    blended = alpha * ov_resized + (1.0 - alpha) * gray3              # [3, H, W]

    # 6. Clamp to [0,255], cast → uint8, move to CPU, return as H×W×3
    blended = blended.clamp(0, 255).round().byte()                    # [3, H, W]
    return blended.permute(1, 2, 0).cpu().numpy()    

def guided_smoothing(rgb: np.ndarray, seg_mask: np.ndarray, radius=4, eps=1e-3) -> np.ndarray:
    rgb_float = rgb.astype(np.float32) / 255.0
    seg_float = seg_mask.astype(np.float32) / 255.0
    guided = cv2.ximgproc.guidedFilter(guide=rgb_float, src=seg_float, radius=radius, eps=eps)
    return (guided * 255).astype(np.uint8)

def superpixel_smoothing(image: np.ndarray, seg_mask: np.ndarray, n_segments=500) -> np.ndarray:
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)
    smoothed = np.zeros_like(seg_mask)
    for label in np.unique(segments):
        mask = segments == label
        values = seg_mask[mask]
        if values.size == 0:
            continue  # skip empty segment
        majority = mode(values, axis=None)[0]
        smoothed[mask] = majority.item()  # safely extract scalar
    return smoothed

def fast_superpixel_seeds(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    num_superpixels = 400  # Adjust as needed
    num_levels = 4
    prior = 2
    histogram_bins = 5
    double_step = False

    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        w, h, image.shape[2],
        num_superpixels, num_levels, prior,
        histogram_bins, double_step
    )

    seeds.iterate(image, num_iterations=2)  # Keep small
    labels = seeds.getLabels()
    out = np.zeros_like(mask)

    for label in np.unique(labels):
        region = mask[labels == label]
        if region.size > 0:
            out[labels == label] = np.bincount(region).argmax()
    return out


def scene_changed(prev: np.ndarray, curr: np.ndarray, threshold=0.02) -> bool:
    # resize if needed
    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))

    # normalize
    prev = prev.astype(np.float32) / 255.0
    curr = curr.astype(np.float32) / 255.0

    diff = np.mean(np.abs(prev - curr))  # Mean absolute difference
    return diff > threshold

def scene_changed_ssim(prev: np.ndarray, curr: np.ndarray, threshold=0.95) -> bool:
    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    score = ssim(prev_gray, curr_gray, data_range=1.0)

    return score < threshold

def compute_ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def warp_mask(prev_rgb, curr_rgb, prev_mask):
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    h, w = prev_mask.shape
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
    remap = flow_map + flow
    warped = cv2.remap(prev_mask, remap[..., 0], remap[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped.astype(np.uint8)

def has_one_large_high_sim_region(
    image: np.ndarray,
    similarity_map: np.ndarray,
    sim_thresh: float = 0.5,
    area_thresh: float = 0.05,
    num_superpixels: int = 500,
    num_levels: int = 3,
    prior: int = 2,
    histogram_bins: int = 5,
    num_iterations: int = 2,
) -> bool:
    """
    Returns True if there exists *one* superpixel whose
    mean similarity ≥ sim_thresh *and* whose size ≥ area_thresh of the image.
    """
    h, w = image.shape[:2]

    # create & run SEEDS to get labels
    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        w, h, image.shape[2],
        num_superpixels, num_levels, prior,
        histogram_bins, False
    )
    seeds.iterate(image, num_iterations=num_iterations)
    labels = seeds.getLabels()

    # check each superpixel individually
    for sp_id in np.unique(labels):
        mask = (labels == sp_id)
        mean_sim = float(similarity_map[mask].mean())
        area_frac = mask.sum() / (h * w)

        if mean_sim >= sim_thresh and area_frac >= area_thresh:
            return True

    return False

def has_one_large_high_sim_region_slic(
    image: np.ndarray,
    similarity_map: np.ndarray,
    sim_thresh: float = 0.5,
    area_thresh: float = 0.05,
    num_superpixels: int = 15,
    compactness: float = 5.0,
    sigma: float = 1.0,
    num_iterations: int = 200,
    boundary_color: tuple = (0, 255, 0),  # BGR green
) -> bool:
    """
    True if ∃ a SLIC superpixel whose mean(similarity_map) ≥ sim_thresh
    and whose area ≥ area_thresh * image_area.
    """
    h, w = image.shape[:2]
    # 1) compute SLIC labels (0…n_segments-1)
    labels = slic(
        image,
        n_segments=num_superpixels,
        compactness=compactness,
        max_num_iter=num_iterations,  
        sigma=sigma,
        start_label=0,
        channel_axis=-1,              # for color images
        convert2lab=False,
        slic_zero=True,
    )
    
    # 2) flatten and bin-count
    flat_lbl = labels.ravel()
    flat_sim = similarity_map.ravel()
    counts   = np.bincount(flat_lbl)
    sums     = np.bincount(flat_lbl, weights=flat_sim)
    
    # 3) mean similarity and area fraction per superpixel
    mean_sim   = sums / counts
    area_frac  = counts / (h * w)
    
    found = bool(np.any((mean_sim >= sim_thresh) & (area_frac >= area_thresh)))

    # 3) build overlay: draw superpixel boundaries on a copy
    boundaries = find_boundaries(labels, mode='outer')
    overlay = image.copy()
    overlay[boundaries] = boundary_color

    return found, overlay

def has_large_high_sim_region_cc(sim_map: np.ndarray,
                                 sim_thresh: float = 0.5,
                                 area_thresh: float = 0.05) -> bool:
    """
    Return True if there is *any* connected component in the binary mask
    (sim_map >= sim_thresh) whose pixel‐count ≥ area_thresh * total_pixels.
    """
    # threshold → binary mask (uint8)
    mask = (sim_map >= sim_thresh).astype(np.uint8)

    H, W = sim_map.shape
    total     = H * W
    min_pixels = int(area_thresh * total)

    sim_score = float(sim_map.max())

    # find connected components *with stats*
    #    stats is an array of shape (num_labels, 5), where
    #    stats[i, cv2.CC_STAT_AREA] is the pixel-count of label i.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    if num_labels <= 1:
        return False, sim_score, 0.0, 0.0

    best_score = 0.0
    area_frac  = 0.0

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < min_pixels:
            continue

        # highest sim within this component
        region_max = float(sim_map[labels == lab].max())

        if region_max > best_score:
            best_score = region_max
            area_frac  = area / total

    found = (best_score >= sim_thresh)
    return found, sim_score, area_frac, best_score
    # # stats[1:, AREA] are the areas of the *foreground* labels 1..L-1
    # areas = stats[1:, cv2.CC_STAT_AREA]
    # max_area = int(areas.max())
    # area_frac = max_area / total

    # # any component big enough?
    # return bool((areas >= min_pixels).any()), sim_score, area_frac

def render_rescale(self, srgb_mono):
    '''This function takes a single channel semantic similarity and rescales it globally'''
    # Maintain running min/max
    if not hasattr(self, "running_min"):
        self.running_min = -1.0
    if not hasattr(self, "running_max"):
        self.running_max = 1.0

    current_min = srgb_mono.min().item()
    current_max = srgb_mono.max().item()
    self.running_min = min(self.running_min, current_min)
    self.running_max = max(self.running_max, current_max)

    similarity_clip = (srgb_mono - self.running_min) / (self.running_max - self.running_min + 1e-10)

    return similarity_clip
