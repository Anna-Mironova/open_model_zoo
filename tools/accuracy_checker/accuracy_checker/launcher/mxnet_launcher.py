"""
Copyright (c) 2019 Intel Corporation

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

import re
from collections import OrderedDict

import numpy as np
import mxnet

from .launcher import Launcher, LauncherConfigValidator, ListInputsField
from ..config import PathField, StringField, NumberField, ConfigError
from ..utils import string_to_tuple

DEVICE_REGEX = r'(?P<device>cpu$|gpu)(_(?P<identifier>\d+))?'


class MxNetLauncherConfigValidator(LauncherConfigValidator):
    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        inputs = entry['inputs']

        for input_layer in inputs:
            if 'shape' not in input_layer:
                raise ConfigError('shape for input {} is not provided'.format(input_layer['name']))


class MxNetLauncher(Launcher):
    """
    Class for infer model using MXNet framework
    """
    __provider__ = 'mxnet'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(check_exists=True, is_directory=False, description="Path to model."),
            'device': StringField(regex=DEVICE_REGEX, description="Device name.", optional=True, default='CPU'),
            'batch': NumberField(value_type=float, min_value=1, optional=True, description="Batch size."),
            'output_name': StringField(optional=True, description="Output name."),
            'inputs': ListInputsField(optional=False, description="Inputs.")
        })
        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)

        mxnet_launcher_config = MxNetLauncherConfigValidator(
            'MxNet_Launcher', fields=self.parameters(), delayed_model_loading=self._delayed_model_loading
        )
        mxnet_launcher_config.validate(self.config)
        if not self._delayed_model_loading:
            # Get model name, prefix, epoch
            self.model = self.config['model']
            model_path, model_file = self.model.parent, self.model.name
            model_name = model_file.rsplit('.', 1)[0]
            model_prefix, model_epoch = model_name.rsplit('-', 1)

            # Get device and set device context
            match = re.match(DEVICE_REGEX, self.config['device'].lower())
            if match.group('device') == 'gpu':
                identifier = match.group('identifier')
                if identifier is None:
                    identifier = 0
                device_context = mxnet.gpu(int(identifier))
            else:
                device_context = mxnet.cpu()

            # Get batch from config or 1
            self._batch = self.config.get('batch', 1)

            # Get input shapes
            input_shapes = []

            for input_config in self.config['inputs']:
                input_shape = input_config['shape']
                input_shape = string_to_tuple(input_shape, casting_type=int)
                input_shapes.append((input_config['name'], (self._batch, *input_shape)))

            # Load checkpoints
            sym, arg_params, aux_params = mxnet.model.load_checkpoint(
                model_path / model_prefix, int(model_epoch)
            )
            self._inputs = OrderedDict(input_shapes)
            # Create a module
            self.module = mxnet.mod.Module(symbol=sym, context=device_context, label_names=None)
            self.module.bind(for_training=False, data_shapes=input_shapes)
            self.module.set_params(arg_params, aux_params, allow_missing=True)

    @property
    def batch(self):
        return self._batch

    def fit_to_input(self, data, input_layer, layout, precision):
        data = np.transpose(data, layout)
        return mxnet.nd.array(data.astype(precision) if precision else data)

    @property
    def inputs(self):
        return self._inputs

    def predict(self, inputs, metadata=None, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """
        results = []
        for infer_input in inputs:
            data_iter = mxnet.io.NDArrayIter(
                data=infer_input, label=None, batch_size=self.batch)
            data_batch = mxnet.io.DataBatch(data=data_iter.data_list)

            # Infer
            self.module.forward(data_batch)
            infer_res = {}
            for layer, out in zip(self.module.output_names, self.module.get_outputs()):
                infer_res[layer.replace('_output', '')] = out.asnumpy()
            results.append(infer_res)

        if metadata is not None:
            for meta_ in metadata:
                meta_['input_shape'] = self.inputs_info_for_meta()

        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError('MXNet Launcher does not support async mode yet')

    @property
    def output_blob(self):
        return self.config.get('output_name', next(iter(self.module.output_names))).replace('_output', '')

    def release(self):
        """
        Releases launcher
        """
        del self.module
