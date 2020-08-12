# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Point cloud network layers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .Pooling import GlobalAveragePooling, GlobalMaxPooling
from .Pooling import MaxPooling, AveragePooling
from .MCConv import MCConv
from .conv1x1 import Conv1x1
from .building_blocks import MCResNet, \
    MCResNetBottleNeck, MCResNetSpatialBottleNeck
from .KPConv import KPConv
from .PointConv import PointConv
from .utils import positional_encoding, spherical_kernel_points
