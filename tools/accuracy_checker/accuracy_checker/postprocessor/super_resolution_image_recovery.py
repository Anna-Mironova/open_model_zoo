"""
Copyright (c) 2018-2020 Intel Corporation

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

import cv2
import numpy as np
from PIL import Image

from ..config import StringField, NumberField
from .postprocessor import Postprocessor
from ..representation import SuperResolutionPrediction, SuperResolutionAnnotation


class SRImageRecovery(Postprocessor):
    __provider__ = 'sr_image_recovery'

    annotation_types = (SuperResolutionAnnotation, )
    prediction_types = (SuperResolutionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'target_color': StringField(optional=True, choices=['bgr', 'rgb'], default='rgb'),
            'input_size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Input size for both dimensions (height and width)"
            )
        })
        return parameters

    def configure(self):
        self.color = cv2.COLOR_YCrCb2BGR if self.get_value_from_config('target_color') == 'bgr' else cv2.COLOR_YCrCb2RGB
        self.input_size = self.get_value_from_config('input_size')

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            data = annotation_.value
            data = Image.fromarray(data, 'RGB')
            data = data.resize((self.input_size, self.input_size), Image.BICUBIC)
            data = np.array(data)
            ycrcbdata = cv2.cvtColor(data, cv2.COLOR_RGB2YCrCb)
            cr = ycrcbdata[:, :, 1]
            cb = ycrcbdata[:, :, 2]
            h, w, _ = prediction_.value.shape
            cr = Image.fromarray(np.uint8(cr), mode='L')
            cb = Image.fromarray(np.uint8(cb), mode='L')
            cr = cr.resize((w, h), Image.BICUBIC)
            cb = cb.resize((w, h), Image.BICUBIC)
            cr = np.expand_dims(np.array(cr).astype(np.uint8), axis=-1)
            cb = np.expand_dims(np.array(cb).astype(np.uint8), axis=-1)
            ycrcb = np.concatenate([prediction_.value, cr, cb], axis=2)
            prediction_.value = cv2.cvtColor(ycrcb, self.color)
        return annotation, prediction
