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
"""Class to represent hierarchical point clouds"""


import numpy as np
import tensorflow as tf

from MCCNN2.pc.utils import check_valid_point_hierarchy_input

from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc import Neighborhood
from MCCNN2.pc import Sample
from MCCNN2.pc import SampleMode


class PointHierarchy:
  """Class to represent a point cloud hierarchy.

  Attributes:
    _aabb (AABB): Bounding box of the point cloud.
    _point_clouds (array of PointCloud): List of point clouds.
    _sample_ops (array of Sample): List of sampling operations used to
      create the point hierarchy.
  """

  def __init__(self,
               point_cloud: PointCloud,
               cell_sizes,
               sample_mode=SampleMode.pd,
               name=None):
    """Constructor.

    Args:
      point_cloud (PointCloud): Input point cloud.
      cell_sizes (array of numpy arrays of floats): List of cell sizes for
        each dimension.
      sample_mode (SampleMode): Mode used to sample the points.
    """
    with tf.compat.v1.name_scope(
        name, "hierarchical point cloud constructor",
        [self, point_cloud, cell_sizes, sample_mode]):

      # check_valid_point_hierarchy_input(point_cloud,cell_sizes,sample_mode)

      #Initialize the attributes.
      self._aabb = point_cloud.get_AABB()
      self._point_clouds = [point_cloud]
      self._sample_ops = []
      self._cell_sizes = []

      self._dimension = point_cloud._dimension
      self._batch_shape = point_cloud._batch_shape

      #Create the different sampling operations.
      cur_point_cloud = point_cloud
      for sample_iter, cur_cell_sizes in enumerate(cell_sizes):
        cur_cell_sizes  = tf.convert_to_tensor(
            value=cur_cell_sizes, dtype=tf.float32)

        # Check if the cell size is defined for all the dimensions.
        # If not, the last cell size value is tiled until all the dimensions
        # have a value.
        cur_num_dims = cur_cell_sizes.shape[0]
        if cur_num_dims < self._dimension:
          cur_cell_sizes = np.concatenate(
              (cur_cell_sizes, np.tile(cur_cell_sizes[-1],
               self._dimension - cur_num_dims)))
        elif cur_num_dims > self._dimension:
          raise ValueError(
              f'Too many dimensions in cell sizes {cur_num_dims} \
                instead of max. {self._dimension}')
        self._cell_sizes.append(cur_cell_sizes)

        #Create the sampling operation.
        cell_sizes_tensor = tf.convert_to_tensor(cur_cell_sizes, np.float32)

        cur_grid = Grid(cur_point_cloud, self._aabb, cell_sizes_tensor)
        cur_neighborhood = Neighborhood(cur_grid, cell_sizes_tensor)
        cur_sample_op = Sample(cur_neighborhood, sample_mode)

        self._sample_ops.append(cur_sample_op)
        cur_sample_op._sample_point_cloud.set_batch_shape(self._batch_shape)
        self._point_clouds.append(cur_sample_op._sample_point_cloud)
        cur_point_cloud = cur_sample_op._sample_point_cloud

  def get_points(self, id=None, max_num_points=None, name=None):
    """ Returns the points.

    Note:
      In the following, A1 to An are optional batch dimensions.

      If called withoud specifying 'id' returns the points in padded format
      [A1,...,An,V,D]

    Args:
      id Identifier of point cloud in the batch, if None return all points

    Return:
      list of tensors:  if 'id' was given: 2D float tensors,
        if 'id' not given: float tensors of shape [A1,...,An,V,D].
    """
    with tf.compat.v1.name_scope(
        name, "get points of specific batch id", [self, id]):
      points = []
      for point_cloud in self._point_clouds:
        points.append(point_cloud.get_points(id))
      return points

  def get_sizes(self, name=None):
    """ Returns the sizes of the point clouds in the point hierarchy.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Return:
      list of tensors of shape [A1,..,An]
    """

    with tf.compat.v1.name_scope(name, "get point hierarchy sizes", [self]):
      sizes = []
      for point_cloud in self._point_clouds:
        sizes.append(point_cloud.get_sizes())
      return sizes

  def set_batch_shape(self, batch_shape, name=None):
    """ Function to change the batch shape

      Use this to set a batch shape instead of using 'self._batch_shape'
      to also change dependent variables.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      batch_shape: float tensor of shape [A1,...,An]

    Raises:
      ValueError: if shape does not sum up to batch size.
    """
    with tf.compat.v1.name_scope(
        name, "set batch shape of point hierarchy", [self, batch_shape]):
      for point_cloud in self._point_clouds:
        point_cloud.set_batch_shape(batch_shape)

  def __getitem__(self, index):
    return self._point_clouds[index]

  def __len__(self):
    return len(self._point_clouds)
