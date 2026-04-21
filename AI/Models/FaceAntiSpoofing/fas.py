"""
Face Anti-Spoofing wrapper around MiniFASNetV2.

Replaces the previous DeePixBiS approach with the lighter, more accurate
MiniFASNetV2 model from:
    https://github.com/minivision-ai/Silent-Face-Anti-Spoofing

Usage:
    from Models.FaceAntiSpoofing import FASModel

    fas = FASModel()
    label, score = fas.predict(frame_bgr, bbox)
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .MiniFASNet import MiniFASNetV2

logger = logging.getLogger(__name__)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(_MODULE_DIR, "2.7_80x80_MiniFASNetV2.pth")

# Model configuration derived from the weight filename:
#   2.7_80x80_MiniFASNetV2.pth  →  scale=2.7, h=80, w=80
_SCALE = 2.7
_H_INPUT = 80
_W_INPUT = 80
_KERNEL = ((_H_INPUT + 15) // 16, (_W_INPUT + 15) // 16)  # (5, 5)


# ── CropImage (from Silent-Face-Anti-Spoofing/src/generate_patches.py) ──

class _CropImage:
    """Create a scaled patch around a face bbox, matching training preprocessing."""

    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2 + x
        center_y = box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return int(left_top_x), int(left_top_y), \
               int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):
        if not crop:
            return cv2.resize(org_img, (out_w, out_h))
        src_h, src_w, _ = np.shape(org_img)
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = \
            self._get_new_box(src_w, src_h, bbox, scale)
        img = org_img[left_top_y: right_bottom_y + 1,
                      left_top_x: right_bottom_x + 1]
        return cv2.resize(img, (out_w, out_h))


_cropper = _CropImage()


class FASModel:
    """MiniFASNetV2 face anti-spoofing model.

    The model outputs a 3-class softmax: class 1 = Real, others = Spoof.
    ``predict()`` returns ``(label, score)`` where *score* is the probability
    of the face being *real* (higher = more likely real).
    """

    THRESHOLD = 0.5

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS_PATH, device: str = "cpu"):
        self.device = torch.device(device)

        self.model = MiniFASNetV2(conv6_kernel=_KERNEL)

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)#weights_only=True to avoid unnecessary optimizer state loading and speed up loading time
        # Handle weights saved with DataParallel ('module.' prefix)
        first_key = next(iter(state_dict))
        if first_key.startswith("module."):
            state_dict = OrderedDict(
                (k[7:], v) for k, v in state_dict.items()
            )
        self.model.load_state_dict(state_dict) #
        self.model.eval().to(self.device)

        logger.info("FASModel (MiniFASNetV2) loaded from %s (device=%s)",
                     weights_path, device)

    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray, bbox: np.ndarray | None = None):
        """
        Run face anti-spoofing inference.

        Args:
            image: Full BGR frame (numpy array).
            bbox:  Face bounding box **[x1, y1, x2, y2]** from the detector.
                   If ``None``, the whole image is treated as a face crop and
                   simply resized to 80×80 (useful for pre-cropped test images).

        Returns:
            ``(label, score)`` where *label* is ``"Real"`` or ``"Spoof"`` and
            *score* ∈ [0, 1] is the real-face probability.
        """
        if bbox is not None:
            # Convert [x1, y1, x2, y2] → [x, y, w, h]
            x1, y1, x2, y2 = bbox.astype(int) if hasattr(bbox, "astype") else list(map(int, bbox))
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            face_patch = _cropper.crop(
                image, bbox_xywh, scale=_SCALE,
                out_w=_W_INPUT, out_h=_H_INPUT, crop=True,
            )
        else:
            # Pre-cropped face — just resize
            face_patch = cv2.resize(image, (_W_INPUT, _H_INPUT))

        tensor = torch.from_numpy(
            face_patch.transpose((2, 0, 1))  # HWC → CHW
        ).float().unsqueeze(0).to(self.device)  # keep [0, 255] — NOT /255

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

        # Class 1 = real face
        score = float(probs[1])
        label = "Real" if score >= self.THRESHOLD else "Spoof"
        return label, score
