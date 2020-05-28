"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import itertools
import math
import numpy as np
import os
import cv2
import re
from collections import namedtuple

class Detector(object):
    def __init__(self, ie, model_path, device='CPU', threshold=0.5):
        model = ie.read_network(model=model_path, weights=os.path.splitext(model_path)[0] + '.bin')

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert (len(model.outputs) == 6 or len(model.outputs) == 9
                or len(model.outputs) == 12), "Expected 6, 9 or 12 output blobs"

        self._input_layer_name = next(iter(model.inputs))
        self._output_layer_names = sorted(model.outputs)

        self._ie = ie
        self._exec_model = self._ie.load_network(model, device)
        self.infer_time = -1
        _, channels, self.input_height, self.input_width = model.inputs[self._input_layer_name].shape
        assert channels == 3, "Expected 3-channel input"

        self.threshold = threshold

        _ratio = (1.,)
        self.anchor_cfg = {
            32: {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio},
            16: {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio},
            8: {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio}
        }
        self._features_stride_fpn = [32, 16, 8]
        self._anchors_fpn = dict(zip(self._features_stride_fpn, self.generate_anchors_fpn(cfg=self.anchor_cfg)))
        self._num_anchors = dict(zip(
            self._features_stride_fpn, [anchors.shape[0] for anchors in self._anchors_fpn.values()]
        ))

        self.bboxes_output = self.create_list_outputs_name(self._output_layer_names, "bbox_pred")
        self.scores_output = self.create_list_outputs_name(self._output_layer_names, "cls_prob")
        self.landmarks_output = self.create_list_outputs_name(self._output_layer_names, "landmark_pred")
        self.type_scores_output = self.create_list_outputs_name(self._output_layer_names, "type_prob")

    @staticmethod
    def create_list_outputs_name(output_names, pattern):
        result = []
        for name in output_names:
            if re.search(pattern, name):
                result.append(name)
        return result

    @staticmethod
    def generate_anchors_fpn(cfg):
        def generate_anchors(base_size=16, ratios=(0.5, 1, 2), scales=2 ** np.arange(3, 6)):
            base_anchor = np.array([1, 1, base_size, base_size]) - 1
            ratio_anchors = _ratio_enum(base_anchor, ratios)
            anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
            return anchors
        def _ratio_enum(anchor, ratios):
            w, h, x_ctr, y_ctr = _generate_wh_ctrs(anchor)
            size = w * h
            size_ratios = size / ratios
            ws = np.round(np.sqrt(size_ratios))
            hs = np.round(ws * ratios)
            anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
            return anchors
        def _scale_enum(anchor, scales):
            w, h, x_ctr, y_ctr = _generate_wh_ctrs(anchor)
            ws = w * scales
            hs = h * scales
            anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
            return anchors
        def _generate_wh_ctrs(anchor):
            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            x_ctr = anchor[0] + 0.5 * (w - 1)
            y_ctr = anchor[1] + 0.5 * (h - 1)
            return w, h, x_ctr, y_ctr
        def _make_anchors(ws, hs, x_ctr, y_ctr):
            ws = ws[:, np.newaxis]
            hs = hs[:, np.newaxis]
            anchors = np.hstack((
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ))
            return anchors

        rpn_feat_stride = [int(k) for k in cfg]
        rpn_feat_stride.sort(reverse=True)
        anchors = []
        for stride in rpn_feat_stride:
            feature_info = cfg[stride]
            bs = feature_info['BASE_SIZE']
            __ratios = np.array(feature_info['RATIOS'])
            __scales = np.array(feature_info['SCALES'])
            anchors.append(generate_anchors(bs, __ratios, __scales))
        return anchors

    def _get_proposals(self, bbox_deltas, anchor_num, anchors):
        bbox_deltas = bbox_deltas.transpose((1, 2, 0))
        bbox_pred_len = bbox_deltas.shape[2] // anchor_num
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        proposals = self.bbox_pred(anchors, bbox_deltas)
        return proposals

    @staticmethod
    def _get_scores(scores, anchor_num):
        scores = scores[anchor_num:, :, :]
        scores = scores.transpose((1, 2, 0)).reshape(-1)
        return scores

    @staticmethod
    def _get_mask_scores(type_scores, anchor_num):
        mask_scores = type_scores[anchor_num * 2:, :, :]
        mask_scores = mask_scores.transpose((1, 2, 0)).reshape(-1)
        return mask_scores

    def _get_landmarks(self, landmark_deltas, anchor_num, anchors):
        landmark_pred_len = landmark_deltas.shape[0] // anchor_num
        landmark_deltas = landmark_deltas.transpose((1, 2, 0)).reshape((-1, 5, landmark_pred_len // 5))
        landmarks = self.landmark_pred(anchors, landmark_deltas)
        return landmarks

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]
        pred_boxes = np.zeros(box_deltas.shape)
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)
        if box_deltas.shape[1] > 4:
            pred_boxes[:, 4:] = box_deltas[:, 4:]
        return pred_boxes

    @staticmethod
    def anchors_plane(height, width, stride, base_anchors):
        num_anchors = base_anchors.shape[0]
        all_anchors = np.zeros((height, width, num_anchors, 4))
        for iw in range(width):
            sw = iw * stride
            for ih in range(height):
                sh = ih * stride
                for k in range(num_anchors):
                    all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                    all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                    all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                    all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh
        return all_anchors

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
            pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
        return pred

    @staticmethod
    def nms(x1, y1, x2, y2, scores, thresh, include_boundaries=True, keep_top_k=None):
        b = 1 if include_boundaries else 0

        areas = (x2 - x1 + b) * (y2 - y1 + b)
        order = scores.argsort()[::-1]

        if keep_top_k:
            order = order[:keep_top_k]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + b)
            h = np.maximum(0.0, yy2 - yy1 + b)
            intersection = w * h

            union = (areas[i] + areas[order[1:]] - intersection)
            overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

            order = order[np.where(overlap <= thresh)[0] + 1]  # pylint: disable=W0143

        return keep

    def preprocess(self, image):
        return cv2.resize(image, (self.input_width, self.input_height))

    def infer(self, image):
        t0 = cv2.getTickCount()
        inputs = {self._input_layer_name: image}
        output = self._exec_model.infer(inputs=inputs)
        self.infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        return output

    def postprocess(self, raw_output, image_sizes):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        mask_scores_list = []
        detection = namedtuple('detection', 'score, x_min, y_min, x_max, y_max')
        landmark = namedtuple('landmark', 'landmark_x_coord, landmark_y_coord')
        detections = []
        res_landmarks = []
        x_scale, y_scale = float(self.input_width) / image_sizes[0], float(self.input_height) /image_sizes[1]
        for _idx, s in enumerate(self._features_stride_fpn):
            anchor_num = self._num_anchors[s]
            scores = self._get_scores(raw_output[self.scores_output[_idx]], anchor_num)
            bbox_deltas = raw_output[self.bboxes_output[_idx]]
            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]
            anchors_fpn = self._anchors_fpn[s]
            anchors = self.anchors_plane(height, width, int(s), anchors_fpn)
            anchors = anchors.reshape((height * width * anchor_num, 4))
            proposals = self._get_proposals(bbox_deltas, anchor_num, anchors)
            mask = scores > self.threshold
            proposals, scores = proposals[mask, :], scores[mask]
            x_mins, y_mins, x_maxs, y_maxs = proposals.T
            if scores.size != 0:
                keep = self.nms(x_mins, y_mins, x_maxs, y_maxs, scores, 0.5, False)
                proposals_list.extend(proposals[keep])
                scores_list.extend(scores[keep])
                if self.type_scores_output:
                    mask_scores = self._get_mask_scores(raw_output[self.type_scores_output[_idx]], anchor_num)[mask]
                    mask_scores_list.extend(mask_scores[keep])
                if self.landmarks_output:
                    landmarks = self._get_landmarks(raw_output[self.landmarks_output[_idx]],
                                                anchor_num, anchors)[mask, :]
                    landmarks_list.extend(landmarks[keep, :])
        scores = np.reshape(scores_list, -1)
        x_mins, y_mins, x_maxs, y_maxs = np.array(proposals_list).T # pylint: disable=E0633

        for score, x_min, y_min, x_max, y_max in zip(scores, x_mins, y_mins, x_maxs, y_maxs):
             detections.append(detection(score=score, x_min=x_min / x_scale, y_min=y_min / y_scale, x_max=x_max / x_scale, y_max=y_max / y_scale))

        if self.landmarks_output:
            landmarks_x_coords = np.array(landmarks_list)[:, :, ::2].reshape(len(landmarks_list), -1) / x_scale
            landmarks_y_coords = np.array(landmarks_list)[:, :, 1::2].reshape(len(landmarks_list), -1) / y_scale
            for landmark_x_coord, landmark_y_coord in zip(landmarks_x_coords, landmarks_y_coords):
                 res_landmarks.append(landmark(landmark_x_coord=landmark_x_coord, landmark_y_coord=landmark_y_coord))

        return detections, mask_scores_list, res_landmarks

        # postprocessing:
        #   - type: cast_to_int
        #   - type: clip_boxes
        #     size: 1024
        #     apply_to: annotation
        #   - type: filter
        #     apply_to: annotation
        #     height_range: 64, 1024
        #     is_empty: True

    def detect(self, image):
        image_sizes = image.shape[:2]
        image = self.preprocess(image)
        image = np.transpose(image, (2, 0, 1))
        output = self.infer(image)
        detections, mask_scores_list, landmarks = self.postprocess({name:output[name][0] for name in self._output_layer_names}, image_sizes)
        return detections, mask_scores_list, landmarks
