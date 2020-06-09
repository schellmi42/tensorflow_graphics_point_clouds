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

import tensorflow as tf
from MCCNN2.pc import PointCloud

def check_valid_point_cloud_input(points,sizes,batch_ids):
    """Cheks that the inputs to the constructor of class 'PointCloud' are valid

    Args:
        points: float tensor of shape [N,D] or [A1,...,An,V,D]
        sizes:  int tensor of shape [A1,...,An] or None
        batch_ids: int tensor of shape [N] or None

    Raises:
        Value Error: If input dimensions are invalid or no valid segmentation is given.
    """

    if sizes is None and batch_ids is None:
        raise ValueError('Missing input! Either sizes or pBatchIds must be given.')
    if len(points.shape)==1:
        raise ValueError('Invalid input! Point tensor is of dimension 1 but should be at least 2!')
    elif len(points.shape)==2 and batch_ids is None:
        raise ValueError('Missing input! No segmentation ids given for input.')

def check_valid_point_hierarchy_input(point_cloud, cell_sizes, pool_mode):
    """ Checks that the inputs to the constructor of class 'PontHierarchy' are valid

    Args:
        point_cloud: an instance of class 'PointCloud'
        cell_sizes: list of float tensors
        pool_mode: int
    
    Raises:
        TypeError: if input is of invalid type
        ValueError: if pool_mode is invalid, or cell_sizes dimension are invalid or non-positive
    """
    if not isinstance(point_cloud,(PointCloud)):
        raise TypeError('Input must be instance of class PointCloud')
    if not pool_mode in [0,1]:
        raise ValueError('Unknown pooling mode.')
    for curr_cell_sizes in cell_sizes:
        if any(curr_cell_sizes <= 0):
            raise ValueError('cell size must be positive.')
        if not len(curr_cell_sizes) in [1,point_cloud.dimension_]:
            raise ValueError('Invalid number of cell sizes for point cloud dimension.  \
                Must be 1 or %s but is %s.'%(point_cloud.dimension_, len(curr_cell_sizes)))