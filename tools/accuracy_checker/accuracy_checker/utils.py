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

import csv
import errno
import itertools
import json
import os
import pickle
from enum import Enum

from pathlib import Path
from typing import Union
from warnings import warn
from collections.abc import MutableSet

from scipy._lib.six import string_types
import zlib
import scipy
import sys

if sys.version_info[0] >= 3:
    byteord = int
else:
    byteord = ord

from numpy.compat import asstr

import numpy as np
import yaml

try:
    import lxml.etree as et
except ImportError:
    import xml.etree.cElementTree as et

try:
    from shapely.geometry.polygon import Polygon
except ImportError:
    Polygon = None

try:
    from yamlloader.ordereddict import Loader as orddict_loader
except ImportError:
    orddict_loader = None


def concat_lists(*lists):
    return list(itertools.chain(*lists))


def get_path(entry: Union[str, Path], is_directory=False, check_exists=True, file_or_directory=False):
    try:
        path = Path(entry)
    except TypeError:
        raise TypeError('"{}" is expected to be a path-like'.format(entry))

    if not check_exists:
        return path

    # pathlib.Path.exists throws an exception in case of broken symlink
    if not os.path.exists(str(path)):
        raise FileNotFoundError('{}: {}'.format(os.strerror(errno.ENOENT), path))

    if not file_or_directory:
        if is_directory and not path.is_dir():
            raise NotADirectoryError('{}: {}'.format(os.strerror(errno.ENOTDIR), path))

        # if it exists it is either file (or valid symlink to file) or directory (or valid symlink to directory)
        if not is_directory and not path.is_file():
            raise IsADirectoryError('{}: {}'.format(os.strerror(errno.EISDIR), path))

    return path


def contains_all(container, *args):
    sequence = set(container)

    for arg in args:
        if len(sequence.intersection(arg)) != len(arg):
            return False

    return True


def contains_any(container, *args):
    sequence = set(container)

    for arg in args:
        if sequence.intersection(arg):
            return True

    return False


def string_to_tuple(string, casting_type=float):
    processed = string.replace(' ', '')
    processed = processed.replace('(', '')
    processed = processed.replace(')', '')
    processed = processed.split(',')

    return tuple([casting_type(entry) for entry in processed]) if not casting_type is None else tuple(processed)


def string_to_list(string):
    processed = string.replace(' ', '')
    processed = processed.replace('[', '')
    processed = processed.replace(']', '')
    processed = processed.split(',')

    return list(entry for entry in processed)


class JSONDecoderWithAutoConversion(json.JSONDecoder):
    """
    Custom json decoder to convert all strings into numbers (int, float) during reading json file.
    """

    def decode(self, s, _w=json.decoder.WHITESPACE.match):
        decoded = super().decode(s, _w)
        return self._decode(decoded)

    def _decode(self, entry):
        if isinstance(entry, str):
            try:
                return int(entry)
            except ValueError:
                pass
            try:
                return float(entry)
            except ValueError:
                pass
        elif isinstance(entry, dict):
            return {self._decode(key): self._decode(value) for key, value in entry.items()}
        elif isinstance(entry, list):
            return [self._decode(value) for value in entry]

        return entry


def dict_subset(dict_, key_subset):
    return {key: value for key, value in dict_.items() if key in key_subset}


def zipped_transform(fn, *iterables, inplace=False):
    result = (iterables if inplace else tuple([] for _ in range(len(iterables))))
    updater = (list.__setitem__ if inplace else lambda container, _, entry: container.append(entry))

    for idx, values in enumerate(zip(*iterables)):
        iter_res = fn(*values)
        if not iter_res:
            continue

        for dst, res in zip(result, iter_res):
            updater(dst, idx, res)

    return result


def overrides(obj, attribute_name, base=None):
    cls = obj if isinstance(obj, type) else obj.__class__

    base = base or cls.__bases__[0]
    obj_attr = getattr(cls, attribute_name, None)
    base_attr = getattr(base, attribute_name, None)

    return obj_attr and obj_attr != base_attr


def enum_values(enum):
    return [member.value for member in enum]


def get_size_from_config(config, allow_none=False):
    if contains_all(config, ('size', 'dst_width', 'dst_height')):
        warn('All parameters: size, dst_width, dst_height are provided. Size will be used. '
             'You should specify only size or pair values des_width, dst_height in config.')
    if 'size' in config:
        return config['size'], config['size']
    if contains_all(config, ('dst_width', 'dst_height')):
        return config['dst_height'], config['dst_width']
    if not allow_none:
        raise ValueError('Either size or dst_width and dst_height required')

    return None, None


def get_size_3d_from_config(config, allow_none=False):
    if contains_all(config, ('size', 'dst_width', 'dst_height', 'dst_volume')):
        warn('All parameters: size, dst_width, dst_height, dst_volume are provided. Size will be used. '
             'You should specify only size or three values des_width, dst_height, dst_volume in config.')
    if 'size' in config:
        return config['size'], config['size'], config['size']
    if contains_all(config, ('dst_width', 'dst_height', 'dst_volume')):
        return config['dst_height'], config['dst_width'], config['dst_volume']
    if not allow_none:
        raise ValueError('Either size or dst_width and dst_height required')

    return config.get('dst_height'), config.get('dst_width'), config.get('dst_volume')


def parse_inputs(inputs_entry):
    inputs = []
    for inp in inputs_entry:
        value = inp.get('value')
        shape = inp.get('shape')
        new_input = {'name': inp['name']}
        if value is not None:
            new_input['value'] = np.array(value) if isinstance(value, list) else value
        if shape is not None:
            new_input['shape'] = shape

        inputs.append(new_input)
    return inputs


def in_interval(value, interval):
    minimum = interval[0]
    maximum = interval[1] if len(interval) >= 2 else None

    if not maximum:
        return minimum <= value

    return minimum <= value < maximum


def is_config_input(input_name, config_inputs):
    for config_input in config_inputs:
        if config_input['name'] == input_name:
            return True
    return False


def finalize_metric_result(values, names):
    result_values, result_names = [], []
    for value, name in zip(values, names):
        if np.isnan(value):
            continue

        result_values.append(value)
        result_names.append(name)

    return result_values, result_names


def get_representations(values, representation_source):
    return np.reshape([value.get(representation_source) for value in values], -1)


def get_supported_representations(container, supported_types):
    if np.shape(container) == ():
        container = [container]

    return list(filter(lambda rep: check_representation_type(rep, supported_types), container))


def check_representation_type(representation, representation_types):
    for representation_type in representation_types:
        if type(representation).__name__ == representation_type.__name__:
            return True
    return False


def is_single_metric_source(source):
    if not source:
        return False

    return np.size(source.split(',')) == 1


def read_txt(file: Union[str, Path], sep='\n', **kwargs):
    def is_empty(string):
        return not string or string.isspace()

    with get_path(file).open(**kwargs) as content:
        content = content.read().split(sep)
        content = list(filter(lambda string: not is_empty(string), content))

        return list(map(str.strip, content))


def read_xml(file: Union[str, Path], *args, **kwargs):
    return et.parse(str(get_path(file)), *args, **kwargs).getroot()


def read_json(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open() as content:
        return json.load(content, *args, **kwargs)


def read_pickle(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open('rb') as content:
        return pickle.load(content, *args, **kwargs)


def read_yaml(file: Union[str, Path], *args, ordered=True, **kwargs):
    with get_path(file).open() as content:
        loader = orddict_loader or yaml.SafeLoader if ordered else yaml.SafeLoader
        if not orddict_loader and ordered:
            warn('yamlloader is not installed. YAML files order is not preserved. it can be sufficient for some cases')
        return yaml.load(content, *args, Loader=loader, **kwargs)


def read_csv(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open() as content:
        return list(csv.DictReader(content, *args, **kwargs))


def extract_image_representations(image_representations):
    images = [rep.data for rep in image_representations]
    meta = [rep.metadata for rep in image_representations]

    return images, meta


def convert_bboxes_xywh_to_x1y1x2y2(x_coord, y_coord, width, height):
    return x_coord, y_coord, x_coord + width, y_coord + height


def get_or_parse_value(item, supported_values=None, default=None, casting_type=float):
    if isinstance(item, str):
        item = item.lower()
        if supported_values and item in supported_values:
            return supported_values[item]

        try:
            return string_to_tuple(item, casting_type=casting_type)
        except ValueError:
            message = 'Invalid value "{}", expected {}list of values'.format(
                item,
                'one of precomputed: ({}) or '.format(', '.join(supported_values.keys())) if supported_values else ''
            )
            raise ValueError(message)

    if isinstance(item, (float, int)):
        return (casting_type(item), )

    if isinstance(item, list):
        return item

    return default


def cast_to_bool(entry):
    if isinstance(entry, str):
        return entry.lower() in ['yes', 'true', 't', '1']
    return bool(entry)


def get_key_by_value(container, target):
    for key, value in container.items():
        if value == target:
            return key

    return None


def format_key(key):
    return '--{}'.format(key)


def to_lower_register(str_list):
    return list(map(lambda item: item.lower() if item else None, str_list))


def polygon_from_points(points):
    if Polygon is None:
        raise ValueError('shapely is not installed, please install it')
    return Polygon(points)


def remove_difficult(difficult, indexes):
    new_difficult = []
    decrementor = 0
    id_difficult = 0
    id_removed = 0
    while id_difficult < len(difficult) and id_removed < len(indexes):
        if difficult[id_difficult] < indexes[id_removed]:
            new_difficult.append(difficult[id_difficult] - decrementor)
            id_difficult += 1
        else:
            decrementor += 1
            id_removed += 1

    return new_difficult


def convert_to_range(entry):
    entry_range = entry
    if isinstance(entry, str):
        entry_range = string_to_tuple(entry_range)
    elif not isinstance(entry_range, tuple) and not isinstance(entry_range, list):
        entry_range = [entry_range]

    return entry_range


def add_input_shape_to_meta(meta, shape):
    meta['input_shape'] = shape
    return meta


def set_image_metadata(annotation, images):
    image_sizes = []
    data = images.data
    if not isinstance(data, list):
        data = [data]
    for image in data:
        data_shape = image.shape if not np.isscalar(image) else 1
        image_sizes.append(data_shape)
    annotation.set_image_size(image_sizes)

    return annotation, images


def get_indexs(container, element):
    return [index for index, container_element in enumerate(container) if container_element == element]


def find_nearest(array, value, mode=None):
    if not array:
        return -1
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if mode == 'less':
        return idx - 1 if array[idx] > value else idx
    if mode == 'more':
        return idx + 1 if array[idx] < value else idx
    return idx


class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev_value, next_value = self.map.pop(key)
            prev_value[2] = next_value
            next_value[1] = prev_value

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '{}()'.format(self.__class__.__name__,)
        return '{}({})'.format(self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


def get_parameter_value_from_config(config, parameters, key):
    if key not in parameters.keys():
        return None
    field = parameters[key]
    value = config.get(key, field.default)
    field.validate(value)
    return value


def check_file_existence(file):
    try:
        get_path(file)
        return True
    except (FileNotFoundError, IsADirectoryError):
        return False


class Color(Enum):
    PASSED = 0
    FAILED = 1


def color_format(s, color=Color.PASSED):
    if color == Color.PASSED:
        return "\x1b[0;32m{}\x1b[0m".format(s)
    return "\x1b[0;31m{}\x1b[0m".format(s)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def loadmat(file_name, mdict=None, appendmat=True, **kwargs):
    variable_names = kwargs.pop('variable_names', None)
    with open_file(file_name, appendmat) as f:
        MR = mat_reader_factory(f, **kwargs)
        matfile_dict = MR.get_variables(variable_names)

    if mdict is not None:
        mdict.update(matfile_dict)
    else:
        mdict = matfile_dict

    return mdict

def open_file(file_like, appendmat, mode='rb'):
    reqs = {'read'} if set(mode) & set('r+') else set()
    if set(mode) & set('wax+'):
        reqs.add('write')
    if reqs.issubset(dir(file_like)):
        return file_like

    try:
        return open(file_like, mode)
    except IOError:
        # Probably "not found"
        if isinstance(file_like, string_types):
            if appendmat and not file_like.endswith('.mat'):
                file_like += '.mat'
            return open(file_like, mode)
        else:
            raise IOError('Reader needs file name or open file-like object')

def mat_reader_factory(file_name, appendmat=True, **kwargs):
    byte_stream = open_file(file_name, appendmat)
    mjv, mnv = get_matfile_version(byte_stream)
    #if mjv == 0:
    #    return MatFile4Reader(byte_stream, **kwargs)
    if mjv == 1:
        return MatFile5Reader(byte_stream, **kwargs)
    elif mjv == 2:
        raise NotImplementedError('Please use HDF reader for matlab v7.3 files')
    else:
        raise TypeError('Did not recognize version %s' % mjv)

def get_matfile_version(fileobj):
    fileobj.seek(0)
    mopt_bytes = fileobj.read(4)
    if len(mopt_bytes) == 0:
        raise ValueError("Mat file appears to be empty")
    mopt_ints = np.ndarray(shape=(4,), dtype=np.uint8, buffer=mopt_bytes)
    if 0 in mopt_ints:
        fileobj.seek(0)
        return (0,0)
    fileobj.seek(124)
    tst_str = fileobj.read(4)
    fileobj.seek(0)
    maj_ind = int(tst_str[2] == b'I'[0])
    maj_val = byteord(tst_str[maj_ind])
    min_val = byteord(tst_str[1-maj_ind])
    ret = (maj_val, min_val)
    if maj_val in (1, 2):
        return ret
    raise ValueError('Unknown mat file type, version %s, %s' % ret)

sys_is_le = sys.byteorder == 'little'
native_code = sys_is_le and '<' or '>'
swapped_code = sys_is_le and '>' or '<'

aliases = {'little': ('little', '<', 'l', 'le'),
           'big': ('big', '>', 'b', 'be'),
           'native': ('native', '='),
           'swapped': ('swapped', 'S')}

class MatFileReader(object):
    def __init__(self, mat_stream,
                 byte_order=None,
                 mat_dtype=False,
                 squeeze_me=False,
                 chars_as_strings=True,
                 matlab_compatible=False,
                 struct_as_record=True,
                 verify_compressed_data_integrity=True
                 ):
        self.mat_stream = mat_stream
        self.dtypes = {}
        if not byte_order:
            byte_order = self.guess_byte_order()
        else:
            byte_order = self.to_numpy_code(byte_order)
        self.byte_order = byte_order
        self.struct_as_record = struct_as_record
        if matlab_compatible:
            self.set_matlab_compatible()
        else:
            self.squeeze_me = squeeze_me
            self.chars_as_strings = chars_as_strings
            self.mat_dtype = mat_dtype
        self.verify_compressed_data_integrity = verify_compressed_data_integrity

    def set_matlab_compatible(self):
        self.mat_dtype = True
        self.squeeze_me = False
        self.chars_as_strings = False

    def guess_byte_order(self):
        return native_code

    def end_of_stream(self):
        b = self.mat_stream.read(1)
        curpos = self.mat_stream.tell()
        self.mat_stream.seek(curpos - 1)
        return len(b) == 0

    def to_numpy_code(code):
        code = code.lower()
        if code is None:
            return native_code
        if code in aliases['little']:
            return '<'
        elif code in aliases['big']:
            return '>'
        elif code in aliases['native']:
            return native_code
        elif code in aliases['swapped']:
            return swapped_code
        else:
            raise ValueError(
                'We cannot handle byte order %s' % code)

mxCELL_CLASS = 1
mxSTRUCT_CLASS = 2
mxOBJECT_CLASS = 3
mxCHAR_CLASS = 4
mxSPARSE_CLASS = 5
mxDOUBLE_CLASS = 6
mxSINGLE_CLASS = 7
mxINT8_CLASS = 8
mxUINT8_CLASS = 9
mxINT16_CLASS = 10
mxUINT16_CLASS = 11
mxINT32_CLASS = 12
mxUINT32_CLASS = 13
mxINT64_CLASS = 14
mxUINT64_CLASS = 15
mxFUNCTION_CLASS = 16
mxOPAQUE_CLASS = 17
mxOBJECT_CLASS_FROM_MATRIX_H = 18

mclass_info = {
    mxINT8_CLASS: 'int8',
    mxUINT8_CLASS: 'uint8',
    mxINT16_CLASS: 'int16',
    mxUINT16_CLASS: 'uint16',
    mxINT32_CLASS: 'int32',
    mxUINT32_CLASS: 'uint32',
    mxINT64_CLASS: 'int64',
    mxUINT64_CLASS: 'uint64',
    mxSINGLE_CLASS: 'single',
    mxDOUBLE_CLASS: 'double',
    mxCELL_CLASS: 'cell',
    mxSTRUCT_CLASS: 'struct',
    mxOBJECT_CLASS: 'object',
    mxCHAR_CLASS: 'char',
    mxSPARSE_CLASS: 'sparse',
    mxFUNCTION_CLASS: 'function',
    mxOPAQUE_CLASS: 'opaque',
    }

def convert_dtypes(dtype_template, order_code):
    dtypes = dtype_template.copy()
    for k in dtypes:
        dtypes[k] = np.dtype(dtypes[k]).newbyteorder(order_code)
    return dtypes

def _convert_codecs(template, byte_order):
    codecs = {}
    postfix = byte_order == '<' and '_le' or '_be'
    for k, v in template.items():
        codec = v['codec']
        try:
            " ".encode(codec)
        except LookupError:
            codecs[k] = None
            continue
        if v['width'] > 1:
            codec += postfix
        codecs[k] = codec
    return codecs.copy()

miINT8 = 1
miUINT8 = 2
miINT16 = 3
miUINT16 = 4
miINT32 = 5
miUINT32 = 6
miSINGLE = 7
miDOUBLE = 9
miINT64 = 12
miUINT64 = 13
miMATRIX = 14
miCOMPRESSED = 15
miUTF8 = 16
miUTF16 = 17
miUTF32 = 18

mdtypes_template = {
    miINT8: 'i1',
    miUINT8: 'u1',
    miINT16: 'i2',
    miUINT16: 'u2',
    miINT32: 'i4',
    miUINT32: 'u4',
    miSINGLE: 'f4',
    miDOUBLE: 'f8',
    miINT64: 'i8',
    miUINT64: 'u8',
    miUTF8: 'u1',
    miUTF16: 'u2',
    miUTF32: 'u4',
    'file_header': [('description', 'S116'),
                    ('subsystem_offset', 'i8'),
                    ('version', 'u2'),
                    ('endian_test', 'S2')],
    'tag_full': [('mdtype', 'u4'), ('byte_count', 'u4')],
    'tag_smalldata':[('byte_count_mdtype', 'u4'), ('data', 'S4')],
    'array_flags': [('data_type', 'u4'),
                    ('byte_count', 'u4'),
                    ('flags_class','u4'),
                    ('nzmax', 'u4')],
    'U1': 'U1',
    }

mclass_dtypes_template = {
    mxINT8_CLASS: 'i1',
    mxUINT8_CLASS: 'u1',
    mxINT16_CLASS: 'i2',
    mxUINT16_CLASS: 'u2',
    mxINT32_CLASS: 'i4',
    mxUINT32_CLASS: 'u4',
    mxINT64_CLASS: 'i8',
    mxUINT64_CLASS: 'u8',
    mxSINGLE_CLASS: 'f4',
    mxDOUBLE_CLASS: 'f8',
    }

codecs_template = {
    miUTF8: {'codec': 'utf_8', 'width': 1},
    miUTF16: {'codec': 'utf_16', 'width': 2},
    miUTF32: {'codec': 'utf_32','width': 4},
    }

def read_dtype(mat_stream, a_dtype):
    num_bytes = a_dtype.itemsize
    arr = np.ndarray(shape=(),
                     dtype=a_dtype,
                     buffer=mat_stream.read(num_bytes),
                     order='F')
    return arr

OPAQUE_DTYPE = np.dtype(
    [('s0', 'O'), ('s1', 'O'), ('s2', 'O'), ('arr', 'O')])

MDTYPES = {}
_N_MIS = 20
_N_MXS = 20
_BLOCK_SIZE = 131072

class GenericStream:
    def __init__(self, fobj):
        self.fobj = fobj

    def seek(self, offset, whence=0):
        self.fobj.seek(offset, whence)
        return 0

    def tell(self):
        return self.fobj.tell()

    def read(self, n_bytes):
        return self.fobj.read(n_bytes)

    def all_data_read(self):
        return 1

    def read_into(self, n):
        count = 0
        p = [0 for i in range(n)]
        while count < n:
            read_size = min(n - count, _BLOCK_SIZE)
            data = self.fobj.read(read_size)
            read_size = len(data)
            if read_size == 0:
                break
            for i in range(read_size):
                p[i] = data[i]
            count += read_size

        if count != n:
            raise IOError('could not read bytes')
        return p, count

def make_stream(fobj):
    if isinstance(fobj, GenericStream):
        return fobj
    return GenericStream(fobj)

class ZlibInputStream(GenericStream):
    def __init__(self, fobj, max_length):
        self.fobj = fobj

        self._max_length = max_length
        self._decompressor = zlib.decompressobj()
        self._buffer = b''
        self._buffer_size = 0
        self._buffer_position = 0
        self._total_position = 0
        self._read_bytes = 0

    def _fill_buffer(self):
        if self._buffer_position < self._buffer_size:
            return

        read_size = min(_BLOCK_SIZE, self._max_length - self._read_bytes)

        block = self.fobj.read(read_size)
        self._read_bytes += len(block)

        self._buffer_position = 0
        if not block:
            self._buffer = self._decompressor.flush()
        else:
            self._buffer = self._decompressor.decompress(block)
        self._buffer_size = len(self._buffer)

    def read_into(self, n):
        dstp = [0 for i in range(n)]
        count = 0
        while count < n:
            self._fill_buffer()
            if self._buffer_size == 0:
                break

            srcp = self._buffer

            size = min(n - count, self._buffer_size - self._buffer_position)
            for i in range (size):
                dstp[i] = srcp[i]
            count += size
            self._buffer_position += size

        return dstp, count

def byteswap_u4(u4):
    return ((u4 << 24) |
           ((u4 << 8) & 0xff0000) |
           ((u4 >> 8 & 0xff00)) |
           (u4 >> 24))

# class VarHeader5:
#     # cdef readonly object name
#     # cdef readonly int mclass
#     # cdef readonly object dims
#     # cdef cnp.int32_t dims_ptr[_MAT_MAXDIMS]
#     # cdef int n_dims
#     # cdef int check_stream_limit
#     # cdef int is_complex
#     # cdef readonly int is_logical
#     # cdef public int is_global
#     # cdef readonly size_t nzmax
#
#     def set_dims(self, dims):
#         self.dims = dims
#         self.n_dims = len(dims)
#         for i, dim in enumerate(dims):
#             self.dims_ptr[i] = int(dim)

_MAT_MAXDIMS = 32

# class VarReader5:
#     def __init__(self, preader):
#         self.dtypes = [None for i in range(_N_MIS)]
#         self.class_dtypes = [None for i in range(_N_MXS)]
#         byte_order = preader.byte_order
#         self.is_swapped = byte_order == swapped_code
#         if self.is_swapped:
#             self.little_endian = not sys_is_le
#         else:
#             self.little_endian = sys_is_le
#         self.struct_as_record = preader.struct_as_record
#         self.codecs = MDTYPES[byte_order]['codecs'].copy()
#         self.uint16_codec = preader.uint16_codec
#         uint16_codec = self.uint16_codec
#         self.codecs['uint16_len'] = len("  ".encode(uint16_codec)) \
#                 - len(" ".encode(uint16_codec))
#         self.codecs['uint16_codec'] = uint16_codec
#         self.cstream = make_stream(preader.mat_stream)
#         self.mat_dtype = preader.mat_dtype
#         self.chars_as_strings = preader.chars_as_strings
#         self.squeeze_me = preader.squeeze_me
#         for key, dt in MDTYPES[byte_order]['dtypes'].items():
#             if isinstance(key, str):
#                 continue
#             self.dtypes[key] = dt
#         for key, dt in MDTYPES[byte_order]['classes'].items():
#             if isinstance(key, str):
#                 continue
#             self.class_dtypes[key] = dt
#
#     def read_full_tag(self):
#         u4s, count = self.cstream.read_into(8)
#         if self.is_swapped:
#             mdtype = byteswap_u4(u4s[0])
#             byte_count = byteswap_u4(count)
#         else:
#             mdtype = u4s[0]
#             byte_count = count
#         return mdtype, byte_count
#
#     def set_stream(self, fobj):
#         self.cstream = make_stream(fobj)
#
#     def cread_tag(self, mdtype_ptr, byte_count_ptr,data_ptr):
#         u4_ptr = data_ptr
#         u4s, count = self.cstream.read_into(8)
#         if self.is_swapped:
#             mdtype = byteswap_u4(u4s[0])
#         else:
#             mdtype = u4s[0]
#         # The most significant two bytes of a U4 *mdtype* will always be
#         # 0, if they are not, this must be SDE format
#         byte_count_sde = mdtype >> 16
#         if byte_count_sde: # small data element format
#             mdtype_sde = mdtype & 0xffff
#             if byte_count_sde > 4:
#                 raise ValueError('Error in SDE format data')
#             u4_ptr[0] = u4s[1]
#             mdtype_ptr[0] = mdtype_sde
#             byte_count_ptr[0] = byte_count_sde
#             return 2
#         if self.is_swapped:
#             byte_count_ptr[0] = byteswap_u4(u4s[1])
#         else:
#             byte_count_ptr[0] = u4s[1]
#         mdtype_ptr[0] = mdtype
#         u4_ptr[0] = 0
#         return 1
#
#     def read_element(self,
#                              mdtype_ptr, byte_count_ptr, pp, copy=True):
#         # cdef cnp.uint32_t mdtype, byte_count
#         # cdef char tag_data[4]
#         # cdef object data
#         # cdef int mod8
#         tag_res = self.cread_tag(mdtype_ptr,
#                                           byte_count_ptr,
#                                           tag_data)
#         mdtype = mdtype_ptr[0]
#         byte_count = byte_count_ptr[0]
#         if tag_res == 1: # full format
#             data = self.cstream.read_string(
#                 byte_count,
#                 pp,
#                 copy)
#             # Seek to next 64-bit boundary
#             mod8 = byte_count % 8
#             if mod8:
#                 self.cstream.seek(8 - mod8, 1)
#         else: # SDE format, make safer home for data
#             data = tag_data[:byte_count]
#             pp[0] = data
#         return data
#
#     def read_element_into(self, ptr, max_byte_count):
#         if max_byte_count < 4:
#             raise ValueError('Unexpected amount of data to read (malformed input file?)')
#         mdtype_ptr= [0]
#         byte_count_ptr = [0]
#         res = self.cread_tag(mdtype_ptr, byte_count_ptr, ptr)
#         byte_count = byte_count_ptr[0]
#         if res == 1: # full format
#             if byte_count > max_byte_count:
#                 raise ValueError('Unexpected amount of data to read (malformed input file?)')
#             res = self.cstream.read_into(ptr, byte_count)
#             mod8 = byte_count % 8
#             if mod8:
#                 self.cstream.seek(8 - mod8, 1)
#         return 0
#
#     def read_int8_string(self):
#         # cdef:
#         #     cnp.uint32_t mdtype, byte_count, i
#         #     void* ptr
#         #     unsigned char* byte_ptr
#         #     object data
#         data = self.read_element(mdtype, byte_count, ptr)
#         if mdtype == miUTF8:
#             byte_ptr = ptr
#             for i in range(byte_count):
#                 if byte_ptr[i] > 127:
#                     raise ValueError('Non ascii int8 string')
#         elif mdtype != miINT8:
#             raise TypeError('Expecting miINT8 as data type')
#         return data
#
#     def read_into_int32s(self, int32p, max_byte_count):
#         mdtype, byte_count = self.read_element_into(int32p, max_byte_count)
#         if mdtype == miUINT32:
#             check_ints = 1
#         elif mdtype != miINT32:
#             raise TypeError('Expecting miINT32 as data type')
#         n_ints = byte_count // 4
#         if self.is_swapped:
#             for i in range(n_ints):
#                 int32p[i] = byteswap_u4(int32p[i])
#         if check_ints:
#             for i in range(n_ints):
#                 if int32p[i] < 0:
#                     raise ValueError('Expecting miINT32, got miUINT32 with '
#                                      'negative values')
#         return n_ints
#
#     def read_header(self, check_stream_limit):
#         u4s, count = self.cstream.read_into(8)
#         if self.is_swapped:
#             flags_class = byteswap_u4(u4s[0])
#             nzmax = byteswap_u4(count)
#         else:
#             flags_class = u4s[0]
#             nzmax = count
#         header = VarHeader5()
#         mc = flags_class & 0xFF
#         header.mclass = mc
#         header.check_stream_limit = check_stream_limit
#         header.is_logical = flags_class >> 9 & 1
#         header.is_global = flags_class >> 10 & 1
#         header.is_complex = flags_class >> 11 & 1
#         header.nzmax = nzmax
#         header.dims_ptr = [0 for i in range(_MAT_MAXDIMS)]
#         if mc == mxOPAQUE_CLASS:
#             header.name = None
#             header.dims = None
#             return header
#         header.n_dims = self.read_into_int32s(header.dims_ptr, len(header.dims_ptr))
#         if header.n_dims > _MAT_MAXDIMS:
#             raise ValueError('Too many dimensions (%d) for numpy arrays'
#                              % header.n_dims)
#         header.dims = [header.dims_ptr[i] for i in range(header.n_dims)]
#         header.name = self.read_int8_string()
#         return header

class MatFile5Reader(MatFileReader):
    def __init__(self,
                 mat_stream,
                 byte_order=None,
                 mat_dtype=False,
                 squeeze_me=False,
                 chars_as_strings=True,
                 matlab_compatible=False,
                 struct_as_record=True,
                 verify_compressed_data_integrity=True,
                 uint16_codec=None
                 ):
        super(MatFile5Reader, self).__init__(
            mat_stream,
            byte_order,
            mat_dtype,
            squeeze_me,
            chars_as_strings,
            matlab_compatible,
            struct_as_record,
            verify_compressed_data_integrity
            )
        if not uint16_codec:
            uint16_codec = sys.getdefaultencoding()
        self.uint16_codec = uint16_codec
        self._file_reader = None
        self._matrix_reader = None

    def guess_byte_order(self):
        self.mat_stream.seek(126)
        mi = self.mat_stream.read(2)
        self.mat_stream.seek(0)
        return mi == b'IM' and '<' or '>'

    def read_file_header(self):
        hdict = {}
        hdr_dtype = MDTYPES[self.byte_order]['dtypes']['file_header']
        hdr = read_dtype(self.mat_stream, hdr_dtype)
        hdict['__header__'] = hdr['description'].item().strip(b' \t\n\000')
        v_major = hdr['version'] >> 8
        v_minor = hdr['version'] & 0xFF
        hdict['__version__'] = '%d.%d' % (v_major, v_minor)
        return hdict

    def initialize_read(self):
        self._file_reader = VarReader5(self)
        self._matrix_reader = VarReader5(self)

    def read_var_header(self):
        mdtype, byte_count = self._file_reader.read_full_tag()
        if not byte_count > 0:
            raise ValueError("Did not read any bytes")
        next_pos = self.mat_stream.tell() + byte_count
        if mdtype == miCOMPRESSED:
            stream = ZlibInputStream(self.mat_stream, byte_count)
            self._matrix_reader.set_stream(stream)
            check_stream_limit = self.verify_compressed_data_integrity
            mdtype, byte_count = self._matrix_reader.read_full_tag()
        else:
            check_stream_limit = False
            self._matrix_reader.set_stream(self.mat_stream)
        if not mdtype == miMATRIX:
            raise TypeError('Expecting miMATRIX type here, got %d' % mdtype)
        header = self._matrix_reader.read_header(check_stream_limit)
        return header, next_pos

    def read_var_array(self, header, process=True):
        return self._matrix_reader.array_from_header(header, process)

    def get_variables(self, variable_names=None):
        if isinstance(variable_names, string_types):
            variable_names = [variable_names]
        elif variable_names is not None:
            variable_names = list(variable_names)

        for _bytecode in '<>':
            _def = {'dtypes': convert_dtypes(mdtypes_template, _bytecode),
                    'classes': convert_dtypes(mclass_dtypes_template, _bytecode),
                    'codecs': _convert_codecs(codecs_template, _bytecode)}
            MDTYPES[_bytecode] = _def

        self.mat_stream.seek(0)
        self.initialize_read()
        mdict = self.read_file_header()
        mdict['__globals__'] = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = asstr(hdr.name)
            if name in mdict:
                warn('Duplicate variable name "%s" in stream'
                              ' - replacing previous with new\n'
                              'Consider mio5.varmats_from_mat to split '
                              'file into single variable files' % name,
                              MatReadWarning, stacklevel=2)
            if name == '':
                name = '__function_workspace__'
                process = False
            else:
                process = True
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            try:
                res = self.read_var_array(hdr, process)
            except MatReadError as err:
                warn(
                    'Unreadable variable "%s", because "%s"' %
                    (name, err),
                    Warning, stacklevel=2)
                res = "Read error: %s" % err
            self.mat_stream.seek(next_position)
            mdict[name] = res
            if hdr.is_global:
                mdict['__globals__'].append(name)
            if variable_names is not None:
                variable_names.remove(name)
                if len(variable_names) == 0:
                    break
        return mdict

    def list_variables(self):
        self.mat_stream.seek(0)
        self.initialize_read()
        self.read_file_header()
        vars = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = asstr(hdr.name)
            if name == '':
                name = '__function_workspace__'

            shape = self._matrix_reader.shape_from_header(hdr)
            if hdr.is_logical:
                info = 'logical'
            else:
                info = mclass_info.get(hdr.mclass, 'unknown')
            vars.append((name, shape, info))

            self.mat_stream.seek(next_position)
        return vars

class MatReadError(Exception):
    pass

class MatReadWarning(UserWarning):
    pass
