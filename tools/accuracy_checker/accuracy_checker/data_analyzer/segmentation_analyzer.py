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

from collections import Counter
from copy import deepcopy
import numpy as np
from .base_data_analyzer import BaseDataAnalyzer
from ..logging import print_info


class SegmentationDataAnalyzer(BaseDataAnalyzer):
    __provider__ = 'SegmentationAnnotation'

    @staticmethod
    def _encode_mask(annotation, segmentation_colors):
        for annotation_ in annotation:
            mask = annotation_.mask.astype(int)
            num_channels = len(mask.shape)
            encoded_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
            for label, color in enumerate(segmentation_colors):
                encoded_mask[np.where(
                    np.all(mask == color, axis=-1) if num_channels >= 3 else mask == color
                )[:2]] = label
            annotation_.mask = encoded_mask.astype(np.int8)

        return annotation

    def analyze(self, result: list, meta, count_objects=True):
        data_analysis = {}
        if count_objects:
            data_analysis['annotations_size'] = self.object_count(result)

        counter = Counter()

        annotations = deepcopy(result)

        for data in annotations:
            data.set_segmentation_mask_source(meta['segmentation_masks_source'])

        segmentation_colors = meta.get('segmentation_colors')
        if segmentation_colors:
            annotations = self._encode_mask(annotations, segmentation_colors)

        for data in annotations:
            for elem in data.mask:
                counter.update(elem)

        label_map = meta.get('label_map', {})
        for key in counter:
            class_name = label_map.get(key, 'class_{key}'.format(key=key))
            print_info('{class_name}: count = {count}'.format(
                class_name=class_name,
                count=counter[key]))
            data_analysis[class_name] = counter[key]

        return data_analysis
