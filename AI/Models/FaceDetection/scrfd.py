"""Local SCRFD ONNX detector implementation.

This module mirrors the inference-time logic used by the official SCRFD reference:
- image letterboxing to model input size
- stride-level decoding of bboxes and keypoints
- confidence thresholding and NMS
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


def distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distances to [x1, y1, x2, y2] boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]

    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])

    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distances to facial keypoints."""
    preds: List[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]

        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])

        preds.append(px)
        preds.append(py)

    return np.stack(preds, axis=-1)


class SCRFD:
    """ONNXRuntime SCRFD detector using official SCRFD-style post-processing."""

    def __init__(
        self,
        model_file: Optional[str] = None,
        session: Optional[ort.InferenceSession] = None,
    ) -> None:
        self.model_file = model_file
        self.session = session
        self.taskname = "detection"
        self.batched = False

        if self.session is None:
            if self.model_file is None:
                raise ValueError("model_file must be provided when session is None")
            if not osp.exists(self.model_file):
                raise FileNotFoundError(f"SCRFD model not found: {self.model_file}")
            self.session = ort.InferenceSession(self.model_file, None)

        self.center_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self) -> None:
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape

        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])

        self.input_name = input_cfg.name
        self.input_shape = input_shape

        outputs = self.session.get_outputs()
        self.batched = len(outputs[0].shape) == 3
        self.output_names = [o.name for o in outputs]

        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1

        outputs_len = len(outputs)
        if outputs_len == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif outputs_len == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif outputs_len == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif outputs_len == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True
        else:
            raise RuntimeError(
                "Unsupported SCRFD output format. "
                f"Expected 6, 9, 10, or 15 outputs, got {outputs_len}."
            )

    def prepare(self, ctx_id: int, **kwargs) -> None:
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])

        nms_thresh = kwargs.get("nms_thresh", None)
        if nms_thresh is not None:
            self.nms_thresh = float(nms_thresh)

        det_thresh = kwargs.get("det_thresh", None)
        if det_thresh is not None:
            self.det_thresh = float(det_thresh)

        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("warning: det_size is already set in scrfd model, ignore")
            else:
                self.input_size = input_size

    def forward(
        self,
        img: np.ndarray,
        threshold: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kpss_list: List[np.ndarray] = []

        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self.fmc][0] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + self.fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self.fmc] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + self.fmc * 2] * stride

            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            if self.use_kps:
                kps_preds = kps_preds.reshape(-1, kps_preds.shape[-1])

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1],
                    axis=-1,
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))

                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors,
                        axis=1,
                    ).reshape((-1, 2))

                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            if pos_inds.size == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds][:, None])
            bboxes_list.append(bboxes[pos_inds])

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    def detect(
        self,
        img: np.ndarray,
        input_size: Optional[Tuple[int, int]] = None,
        max_num: int = 0,
        metric: str = "default",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if input_size is None and self.input_size is None:
            raise ValueError("input_size must be provided when model has dynamic size")
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]

        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        if len(scores_list) == 0 or len(bboxes_list) == 0:
            empty_det = np.empty((0, 5), dtype=np.float32)
            empty_kps = np.empty((0, 5, 2), dtype=np.float32) if self.use_kps else None
            return empty_det, empty_kps

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale if self.use_kps else None

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        if self.use_kps and kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)

            if metric == "max":
                values = area
            else:
                values = area - offset_dist_squared * 2.0

            bindex = np.argsort(values)[::-1]
            bindex = bindex[:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss

    def nms(self, dets: np.ndarray) -> List[int]:
        thresh = self.nms_thresh

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


def get_scrfd(name: str, download: bool = False, **kwargs):
    """Compatibility helper: local files only, no remote download support."""
    if download:
        raise RuntimeError(
            "download=True is not supported in local SCRFD mode. "
            "Provide a local ONNX path instead."
        )
    if not os.path.exists(name):
        raise FileNotFoundError(f"SCRFD model file not found: {name}")
    return SCRFD(name, **kwargs)
