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
"""Class to represent a point cloud distributed in a regular grid.

Attributes:
  _batch_size (int): Size of the batch.
  _cell_sizes (float tensor d): Cell size.
  _point_cloud (PointCloud): Point cloud.
  _aabb (AABB): AABB.
  _num_cells (int tensor d): Number of cells of the grids.
  _cur_keys (int tensor n): Keys of each point.
  _sorted_keys (int tensor n): Keys of each point sorted.
  _sorted_points (float tensor nxd):the sorted points.
  _sorted_indices (int tensor n): Original indices to the original
    points.
  _fast_DS (int tensor BxCXxCY): Fast access data structure.
"""


import tensorflow as tf

from MCCNN2.pc.custom_ops import compute_keys, build_grid_ds
from MCCNN2.pc import PointCloud, AABB


class Grid:
  """ 2D regular grid of a point cloud.

  Args:
    point_cloud : A `PointCloud` instance to distribute in the grid.
    cell_sizes A `float` tensor of shape `[D]`, the sizes of the grid
      cells in each dimension.
    aabb: An `AABB` instance, the bounding box of the grid, if `None`
      the bounding box of `point_cloud` is used. (optional)
  """

  def __init__(self, point_cloud: PointCloud, cell_sizes, aabb=None,
               name=None):
    with tf.compat.v1.name_scope(
        name, "constructor for point cloud regular grid",
        [self, point_cloud, aabb, cell_sizes]):
      cell_sizes = tf.convert_to_tensor(value=cell_sizes, dtype=tf.float32)
      if False:  # cell_sizes in point_cloud._grid_cache:
        # load from memory
        self = point_cloud._grid_cache[cell_sizes]
      else:
        #Save the attributes.
        self._batch_size = point_cloud._batch_size
        self._cell_sizes = cell_sizes
        self._point_cloud = point_cloud
        self._aabb = point_cloud.get_AABB()

        #Compute the number of cells in the grid.
        aabb_sizes = self._aabb._aabb_max - self._aabb._aabb_min
        batch_num_cells = tf.cast(
            tf.math.ceil(aabb_sizes / self._cell_sizes), tf.int32)
        self._num_cells = tf.maximum(
            tf.reduce_max(batch_num_cells, axis=0), 1)

        #Compute the key for each point.
        self._cur_keys = compute_keys(
            self._point_cloud, self._num_cells,
            self._cell_sizes)

        #Sort the keys.
        self._sorted_indices = tf.argsort(
            self._cur_keys, direction='DESCENDING')
        self._sorted_keys = tf.gather(self._cur_keys, self._sorted_indices)

        #Compute the invert indexs.
        # self.invertedIndices_ = tf.argsort(self._sorted_indices)

        #Get the sorted points and batch ids.
        self._sorted_points = tf.gather(
            self._point_cloud._points, self._sorted_indices)
        self._sorted_batch_ids = tf.gather(
            self._point_cloud._batch_ids, self._sorted_indices)

        #Build the fast access data structure.
        self._fast_DS = build_grid_ds(
            self._sorted_keys, self._num_cells, self._batch_size)

        # add grid to the cache
        # point_cloud._grid_cache[hash(cell_sizes)] = self
