import time
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

if not hasattr(torch, "get_default_device"):
    def _compat_get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = _compat_get_default_device

import warnings

try:
    import timm
    # try to access timm.data - older timm may still have this module but lack the API
    try:
        import timm.data as _timm_data
    except Exception:
        # create a small module-like container if timm.data import fails
        import types
        _timm_data = types.SimpleNamespace()
        timm.data = _timm_data

    # Provide ImageNetInfo class if missing
    if not hasattr(_timm_data, "ImageNetInfo"):
        class ImageNetInfo:
            """
            Minimal compatibility shim for timm.data.ImageNetInfo expected by transformers' timm wrapper.
            This contains only very small metadata used in some model wrappers.
            """
            def __init__(self):
                self.num_classes = 1000
                # common imagenet mean/std used by many models
                self.mean = (0.485, 0.456, 0.406)
                self.std = (0.229, 0.224, 0.225)
                # placeholder for other potential attributes
                self.class_to_idx = None
                self.label = None

            def __repr__(self):
                return f"ImageNetInfo(num_classes={self.num_classes})"

        setattr(_timm_data, "ImageNetInfo", ImageNetInfo)
        warnings.warn("timm.data.ImageNetInfo was missing — patched a minimal shim. "
                      "This enables transformers Grounding DINO loading without upgrading timm.")

    # Provide infer_imagenet_subset if missing
    if not hasattr(_timm_data, "infer_imagenet_subset"):
        def infer_imagenet_subset(*args, **kwargs):
            """
            Minimal no-op implementation. transformers may call this to
            check for dataset variants (e.g., 'imagenet2012' subsets). Returning None
            signals 'unknown' and typically doesn't break model instantiation.
            """
            return None

        setattr(_timm_data, "infer_imagenet_subset", infer_imagenet_subset)

except ImportError:
    # timm is not installed; leave transformers to raise the usual error
    warnings.warn("timm not installed in environment. If Grounding DINO requires timm, "
                  "install a compatible timm or provide a shim.")

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamProcessor,
    SamModel,
)

# ---------- utilities (drop-in minimal replacements for your helpers) ----------

def _ensure_rgb_np(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    # numpy
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] == 3:
        return img
    if img.shape[2] == 4:
        return img[:, :, :3]
    raise TypeError("Unsupported numpy image format")

def _to_pil(img: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if img.ndim == 2:
        return Image.fromarray(img).convert("RGB")
    if img.shape[2] == 3:
        return Image.fromarray(img)
    if img.shape[2] == 4:
        return Image.fromarray(img[:, :, :3])
    raise TypeError("Unsupported image format")

def _blend_overlay(image_rgb: np.ndarray, color_mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    # alpha blend: image * 1.0 + color_mask * alpha
    return cv2.addWeighted(image_rgb, 1.0, color_mask_rgb, alpha, 0.0)

def _colorize_soft_mask(prob: np.ndarray) -> np.ndarray:
    """
    Simple viridis-like color without relying on matplotlib.
    Expects prob in [0,1], returns uint8 RGB.
    """
    x = np.clip(prob, 0.0, 1.0)
    # quick colormap: grayscale → RGB (you can plug your LUT here if desired)
    x255 = (x * 255).astype(np.uint8)
    color = cv2.applyColorMap(x255, cv2.COLORMAP_TURBO)  # RGB after cvt
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

@dataclass
class InstanceSeg:
    label: str
    det_score: float
    box_xyxy: Tuple[float, float, float, float]
    iou_score: float
    mask_prob: np.ndarray          # (H, W) float32 in [0,1]
    mask_u8: np.ndarray            # (H, W) uint8 {0,255}
    patch_bgr: Optional[np.ndarray]

# ---------- main class ----------

class GroundedSAMHFModel:
    """
    Grounding DINO (text-conditioned boxes) → SAM (masks).
    API mirrors CLIPSegHFModel.clipseg_hf_inference as closely as possible.

    Returns:
        overlayed: RGB image with colored overlay (uint8 HxWx3)
        scaled: float64/float32 soft map in [0,1] (aggregated max over instances)
        present: bool flag if any instance satisfies thresholds
        extras: dict with "instances" (list[InstanceSeg]) and "instance_map" (int32 id map)
    """

    def __init__(
        self,
        gd_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam-vit-base",
        device: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        mask_iou_threshold: float = 0.0,     # keep all SAM proposals by default
        overlay_alpha: float = 0.45,
        return_patches: bool = True,
    ):
        self.device = torch.device(device) if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.mask_iou_threshold = mask_iou_threshold
        self.overlay_alpha = overlay_alpha
        self.return_patches = return_patches

    # ==== LOITER state (mirrors your CLIPSeg model) ====
        self.lut = None  # placeholder if you want to reuse your color LUT later

        # Running references for calibration
        self.loiter_max = 0.0                  # best per-frame peak found so far
        self.loiter_area_frac = 0.0            # area fraction of best region

        # Stored reference region (from calibration phase)
        self.loiter_mask = None                # uint8 0/255
        self.loiter_cnt = None                 # reference contour
        self.loiter_solidity = None
        self.loiter_eccentricity = None

        # Matching thresholds (tune to taste)
        self.shape_thresh = 0.20               # lower = stricter (cv2.matchShapes I1 distance)
        self.area_tolerance = 0.15             # ±15%
        self.sol_tol = 0.10                    # ±10%
        self.ecc_tol = 0.15                    # ±15%

        # Overlay style
        self.overlay_color = (0, 255, 0)       # RGB (green for calibration; active switches to red)
        # Optional debug cache
        self._loiter_dbg = {}
    #
        # Grounding DINO
        self.gd_processor = AutoProcessor.from_pretrained(gd_model_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(self.device).eval()

        # SAM
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device).eval()

    # ----- internal helpers -----

    def _boxes_from_grounding_dino(self, image_pil: Image.Image, prompts: Union[str, List[str]]):
        # Normalize prompts to [["class1", "class2", ...]]
        if isinstance(prompts, str):
            toks = [p.strip() for p in prompts.split(".") if p.strip()]
            text_labels = [toks if toks else [prompts]]
        else:
            text_labels = [prompts]

        inputs = self.gd_processor(images=image_pil, text=text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)

        results = self.gd_processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image_pil.size[::-1]],  # (H, W)
        )
        r = results[0]
        boxes = r["boxes"].detach().cpu().numpy().tolist()    # xyxy in pixels
        scores = r["scores"].detach().cpu().numpy().tolist()
        labels = r["labels"]
        return boxes, scores, labels

    def _sam_masks_soft(
        self,
        image_pil: Image.Image,
        boxes_xyxy: List[List[float]],
        out_size_hw: Tuple[int, int],
    ):
        """
        Returns:
        masks_prob: List[np.ndarray(H,W) float32 in [0,1]]
        masks_u8:   List[np.ndarray(H,W) uint8 {0,255}]
        iou_scores: List[float]
        """
        import torch
        import torch.nn.functional as F

        if not boxes_xyxy:
            return [], [], []

        # 1) Run SAM
        inputs = self.sam_processor(image_pil, input_boxes=[boxes_xyxy], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        pred = outputs.pred_masks  # various shapes depending on version/checkpoint
        # Expected most commonly: [B, N, 1, Hs, Ws] OR [B, N, Hs, Ws]
        # But we’ll normalize robustly.

        # Helper: ensure we know N (number of boxes)
        num_boxes = len(boxes_xyxy)

        # 2) Convert logits -> prob and rearrange to [B*N, 1, Hs, Ws]
        if isinstance(pred, torch.Tensor):
            t = pred
        else:
            # Very unlikely, but just in case
            t = torch.as_tensor(pred, device=self.device)

        # Sigmoid first
        t = torch.sigmoid(t)

        # Normalize dims
        # Cases:
        #  - 5D: [B, N, 1, Hs, Ws] or [B, 1, Hs, Ws, N] or [B, C, Hs, Ws, N]
        #  - 4D: [B, N, Hs, Ws] or [N, C, Hs, Ws] (batch collapsed)
        #  - 3D: [N, Hs, Ws]
        if t.dim() == 5:
            B, D1, D2, Hs, Ws = t.shape
            # Try to detect where N (num_boxes) is
            # Common: [B, N, 1, Hs, Ws]
            if D2 == 1:
                # [B, N, 1, Hs, Ws] -> [B*N, 1, Hs, Ws]
                B, N, C, Hs, Ws = t.shape
                t = t.view(B * N, C, Hs, Ws)
            # Alternate: [B, 1, Hs, Ws, N] -> permute to [B, N, 1, Hs, Ws] -> [B*N, 1, Hs, Ws]
            elif t.shape[-1] == num_boxes:
                # [B, C, Hs, Ws, N] or [B, 1, Hs, Ws, N]
                # Move N to dim=1 and C to dim=2 if needed
                # Current order: (0=B, 1=C, 2=Hs, 3=Ws, 4=N)
                t = t.permute(0, 4, 1, 2, 3)  # [B, N, C, Hs, Ws]
                B, N, C, Hs, Ws = t.shape
                t = t.view(B * N, C, Hs, Ws)
            else:
                # Fallback: assume [B, N, C, Hs, Ws]
                B, N, C, Hs, Ws = t.shape
                t = t.view(B * N, C, Hs, Ws)

        elif t.dim() == 4:
            # Could be [B, N, Hs, Ws]  -> add channel
            # Or [N, C, Hs, Ws]       -> treat N as batch already
            B, C_orN, Hs, Ws = t.shape
            # Heuristic: if C_orN == num_boxes and B == 1, assume [B=1, N, Hs, Ws]
            if B == 1 and C_orN == num_boxes:
                t = t.view(num_boxes, 1, Hs, Ws)  # [N, 1, Hs, Ws]
            else:
                # Assume it's [B, C, Hs, Ws] or [N, C, Hs, Ws]; ensure channel exists
                if C_orN == 1:
                    t = t.view(B * 1, 1, Hs, Ws)  # [B, 1, Hs, Ws]
                else:
                    # If channel != 1, keep it; we’ll reduce later per-instance by max across channel
                    t = t  # [B, C, Hs, Ws]
            # If it’s [N, 1, Hs, Ws] or [B, 1, Hs, Ws], we’re fine

        elif t.dim() == 3:
            # [N, Hs, Ws] -> add channel
            N, Hs, Ws = t.shape
            t = t.view(N, 1, Hs, Ws)  # [N, 1, Hs, Ws]

        else:
            raise ValueError(f"Unexpected pred_masks shape: {tuple(t.shape)}")

        # Ensure we have a channel dim
        if t.dim() == 3:
            t = t.unsqueeze(1)  # [*, 1, Hs, Ws]

        # 3) Resize to original image size
        H, W = out_size_hw
        t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)  # [M, C, H, W]
        M, C, H, W = t.shape

        # If C>1 (unexpected), collapse channel by max
        if C > 1:
            t = t.max(dim=1, keepdim=True).values  # [M, 1, H, W]

        # 4) Split back into per-instance [H, W]
        # We don't know if M equals N or B*N, but either way each entry is one mask.
        prob_np = t.squeeze(1).detach().float().cpu().numpy()  # [M, H, W]
        masks_prob = [prob_np[i] for i in range(prob_np.shape[0])]
        masks_u8 = [(m > 0.5).astype(np.uint8) * 255 for m in masks_prob]

        # 5) IoU scores: flatten safely
        iou = outputs.iou_scores
        if isinstance(iou, torch.Tensor):
            iou_np = iou.detach().cpu().numpy().reshape(-1).tolist()
        else:
            iou_np = list(iou)

        return masks_prob, masks_u8, iou_np

    def _crop_patch(self, image_rgb: np.ndarray, mask_u8: np.ndarray, box_xyxy: List[float]) -> np.ndarray:
        x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image_rgb.shape[1], x1), min(image_rgb.shape[0], y1)
        crop = image_rgb[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return None
        m = (mask_u8[y0:y1, x0:x1] > 0).astype(np.uint8)
        bg = np.ones_like(crop, dtype=np.uint8) * 255
        crop = np.where(m[..., None] == 1, crop, bg)
        return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    # ----- public API (mirrors CLIPSeg) -----

    def grounded_sam_hf_inference(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: Union[str, List[str]],
        resize_output_to_input: bool = True,   # kept for signature parity (SAM already outputs full size)
        use_refinement: bool = False,          # not used; kept for parity
        use_smoothing: bool = False,           # not used; kept for parity
        scene_change_threshold: float = 1.00,  # not used; kept for parity
        verbose: bool = False,
    ):
        """
        Returns:
            overlayed (HxWx3 uint8): colored overlay on the original image
            scaled (HxW float32): per-pixel soft mask in [0,1] (max over instances)
            present (bool): True if any query instance detected/segmented
            extras (dict): {
                "instances": List[InstanceSeg],
                "instance_map": int32 id map (0 bg, k=index+1)
            }
        """
        def log(*a):
            if verbose:
                print(*a)

        img_rgb = _ensure_rgb_np(image)
        img_pil = _to_pil(image)
        H, W = img_rgb.shape[:2]

        t0 = time.time()
        # 1) Text-conditioned detection
        boxes, scores, labels = self._boxes_from_grounding_dino(img_pil, prompt)
        log(f"GroundingDINO: {len(boxes)} boxes")

        # Early-out if nothing detected
        if len(boxes) == 0:
            overlayed = img_rgb.copy()
            scaled = np.zeros((H, W), dtype=np.float32)
            present = False
            extras = {"instances": [], "instance_map": np.zeros((H, W), dtype=np.int32)}
            return overlayed, scaled, present, extras

        # 2) SAM → soft masks (probabilities in [0,1])
        masks_prob, masks_u8, ious = self._sam_masks_soft(img_pil, boxes, (H, W))

        # 3) Build per-instance data and aggregated map
        instances: List[InstanceSeg] = []
        instance_map = np.zeros((H, W), dtype=np.int32)

        # Aggregate a single soft map similar to CLIPSeg output.
        # We weight each mask by its detection score to keep a rough probabilistic interpretation.
        agg = np.zeros((H, W), dtype=np.float32)

        for i, (box, lbl, det_s) in enumerate(zip(boxes, labels, scores)):
            iou_s = ious[i] if i < len(ious) else 0.0
            if iou_s < self.mask_iou_threshold:
                continue
            prob = masks_prob[i].astype(np.float32)           # [H,W] in [0,1]
            mask_u8 = masks_u8[i]
            # Combine by pixelwise max of (prob * det_score)
            agg = np.maximum(agg, prob * float(det_s))

            patch_bgr = self._crop_patch(img_rgb, mask_u8, box) if self.return_patches else None
            instances.append(
                InstanceSeg(
                    label=lbl,
                    det_score=float(det_s),
                    box_xyxy=tuple(box),
                    iou_score=float(iou_s),
                    mask_prob=prob,
                    mask_u8=mask_u8,
                    patch_bgr=patch_bgr,
                )
            )
            instance_map[mask_u8 > 0] = i + 1

        # presence flag: any kept instance
        present = len(instances) > 0

        # 4) Create overlay similar to CLIPSeg colorized + blend
        colorized = _colorize_soft_mask(agg)               # RGB
        overlayed = _blend_overlay(img_rgb, colorized, alpha=self.overlay_alpha)

        t1 = time.time()
        log(f"Grounded SAM inference: {t1 - t0:.3f}s, instances kept: {len(instances)}")

        # 'scaled' is the soft map akin to CLIPSeg's normalized output
        scaled = agg  # already in [0,1], proportional to prob * det_score

        extras = {"instances": instances, "instance_map": instance_map}
        return overlayed, scaled, present, extras
    
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
            l1 = float(eigenvalues[0][0])
            l2 = float(eigenvalues[1][0]) if eigenvalues.shape[0] > 1 else 1.0
            eccentricity = (1.0 - (min(l1, l2) / max(l1, l2))) if max(l1, l2) > 0 else 0.0
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
        corresponding quantile of the 'logits' array (here: softmap ∈ [0,1]).
        Returns 0/255 uint8 mask.
        """
        H, W = logits.shape
        target_frac = float(np.clip(target_frac, 1e-4, 0.90))  # safety
        q = 1.0 - target_frac
        t = float(np.quantile(logits, q))
        mask = (logits >= t).astype(np.uint8) * 255

        if do_open_close:
            kernel = np.ones((ksize, ksize), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    def _make_overlay(self, frame_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """
        Alpha-blend a solid color (self.overlay_color) where mask==255 over the RGB frame.
        Returns RGB.
        """
        overlay = frame_rgb.copy()
        fill = np.zeros_like(overlay, dtype=np.uint8)
        # self.overlay_color is RGB; OpenCV addWeighted works fine on RGB arrays too
        fill[mask_u8 > 0] = self.overlay_color
        return cv2.addWeighted(overlay, 1.0, fill, 0.40, 0.0)

    def _dbg(self, **k):
        self._loiter_dbg.update(k)
        # print if you like:
        # print("[LOITER DBG]", " ".join(f"{kk}={vv}" for kk, vv in k.items()))

    def loiter_calibrate(
        self,
        logits: np.ndarray,
        frame_img: np.ndarray,
        active_arm: bool = False
    ):
        """
        Returns (found, sim_score, area_frac, overlay_rgb)
          - found: whether current shape matches stored reference (only True in active_arm=True mode)
          - sim_score: global max in logits (soft map)
          - area_frac: area fraction of the current best region (by peak)
          - overlay_rgb: frame with current region and/or stored outline overlaid (RGB)
        """
        found = False
        H, W = logits.shape
        total_area = float(H * W)
        sim_score = float(logits.max())

        # 1) coarse threshold at 90th percentile → connected components
        thresh = float(np.percentile(logits, 90.0))
        cc_mask = (logits >= thresh).astype(np.uint8)  # 0/1
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cc_mask, connectivity=8)
        if num_labels <= 1:
            # Nothing prominent this frame
            return found, sim_score, 0.0, None

        # 2) pick region with highest peak (break ties by area)
        best_lab = -1
        best_region_max = -1.0
        best_area_px = -1
        for lab in range(1, num_labels):
            area_px = stats[lab, cv2.CC_STAT_AREA]
            region_max = float(logits[labels == lab].max())
            if (region_max > best_region_max) or (np.isclose(region_max, best_region_max) and area_px > best_area_px):
                best_region_max = region_max
                best_area_px = area_px
                best_lab = lab

        # area fraction of the "best" region (used to form a cleaner mask)
        area_frac = best_area_px / total_area
        curr_region_mask = self._area_targeted_mask(logits, target_frac=area_frac)

        # color for overlay (green in calibration, red in active)
        self.overlay_color = (0, 255, 0) if not active_arm else (255, 0, 0)
        overlay = self._make_overlay(frame_img, curr_region_mask)

        if not active_arm:
            # === CALIBRATION PHASE ===
            cnt, area_px, solidity, ecc = self._largest_contour_from_mask(curr_region_mask)
            better = (best_region_max > self.loiter_max) or (
                np.isclose(best_region_max, self.loiter_max, rtol=0, atol=1e-6)
                and (best_area_px > self.loiter_area_frac * total_area)
            )
            if better and cnt is not None:
                # store reference
                self.loiter_max = best_region_max
                self.loiter_area_frac = area_frac
                self.loiter_mask = curr_region_mask.copy()
                self.loiter_cnt = cnt
                self.loiter_solidity = solidity
                self.loiter_eccentricity = ecc

                # draw stored outline for debugging
                cv2.drawContours(overlay, [self.loiter_cnt], -1, (0, 200, 255), 3)

            # keep some telemetry
            self._dbg(phase="calib", peak=best_region_max, area_frac=area_frac)
            return found, sim_score, area_frac, overlay

        # === ACTIVE / ARM PHASE ===
        if self.loiter_cnt is None:
            # No reference yet—nothing to match
            self._dbg(phase="active", reason="no_ref")
            return found, sim_score, area_frac, overlay

        # Rebuild mask to match the *reference* area fraction for shape comparison
        curr_region_mask = self._area_targeted_mask(logits, target_frac=self.loiter_area_frac)
        cur_area_frac = float(np.count_nonzero(curr_region_mask)) / total_area

        cur_cnt, cur_area_px, cur_sol, cur_ecc = self._largest_contour_from_mask(curr_region_mask)
        if cur_cnt is None:
            self._dbg(phase="active", reason="no_cnt")
            return found, sim_score, cur_area_frac, overlay

        # 1) Area band
        area_ok = abs(cur_area_frac - self.loiter_area_frac) <= self.area_tolerance * max(self.loiter_area_frac, 1e-6)
        # 2) Shape distance (lower better)
        d = self._match_shape_distance(self.loiter_cnt, cur_cnt)
        shape_ok = (d <= self.shape_thresh)
        # 3) Morphology bands
        sol_ok = (abs(cur_sol - self.loiter_solidity) <= self.sol_tol * max(self.loiter_solidity, 1e-6))
        ecc_ok = (abs(cur_ecc - self.loiter_eccentricity) <= self.ecc_tol * max(self.loiter_eccentricity, 1e-6))

        all_ok = (shape_ok and area_ok and sol_ok and ecc_ok)
        self._dbg(phase="active", d=d, area_ok=area_ok, shape_ok=shape_ok, sol_ok=sol_ok, ecc_ok=ecc_ok)

        # If matched, draw both contours (ref in orange, current in green)
        if all_ok:
            found = True
            rgb_overlay = frame_img.copy()
            fill = np.zeros_like(rgb_overlay); fill[curr_region_mask > 0] = (0, 255, 0)
            rgb_overlay = cv2.addWeighted(fill, 0.4, rgb_overlay, 1.0, 0.0)
            cv2.drawContours(rgb_overlay, [self.loiter_cnt], -1, (255, 165, 0), 2)  # orange in RGB
            cv2.drawContours(rgb_overlay, [cur_cnt], -1, (0, 255, 0), 2)
            overlay = rgb_overlay

        return found, sim_score, area_frac, overlay