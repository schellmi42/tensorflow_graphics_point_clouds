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
"""Class to represent a neighborhood of points.

Note:
  In the following D is the spatial dimensionality of the points,
  N is the number of (samples) points, and M is the total number of
  adjacencies.

Attributes:
  _point_cloud_sampled: 'PointCloud', samples point cloud.
  _grid : 'Grid', regular grid data structure.
  _radii: float 'Tensor' of shape [D], radii used to select the neighbors.
  _samples_neigh_ranges: int 'Tensor' of shape [N], end of the ranges per
    sample.
  _neighbors: int 'Tensor' of shape [M,2], indices of the neighbor point and
    the sample for each neighbor.
  _pdf: float 'Tensor' of shape [M], PDF value for each neighbor.
"""

import enum
import tensorflow as tf

from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc.custom_ops import find_neighbors, compute_pdf


class KDEMode(enum.Enum):
  """ Parameters for kernel density estimation (KDE) """
  constant = 0
  num_points = 1
  no_pdf = 2


class Neighborhood:
  """ Neighborhood of a point cloud.

  Args:
    grid: A 'Grid' instance, the regular grid data structure.
    radii: A float 'Tensor' of shape [D], the radii used to select the
      neighbors.
    point_cloud_sample: A 'PointCloud' instance. Samples point cloud.
      If None, the sorted points from the grid will be used.
    max_neighbors: An `int, maximum number of neighbors per sample,
      if `0` all neighbors are selected.
  """

  def __init__(self,
               grid: Grid,
               radii,
               point_cloud_sample=None,
               max_neighbors=0,
               name=None):
    with tf.compat.v1.name_scope(
        name, "constructor for neighbourhoods of point clouds",
        [self, grid, radii, point_cloud_sample, max_neighbors]):
      radii = tf.convert_to_tensor(value=radii, dtype=tf.float32)

      #Save the attributes.
      if point_cloud_sample is None:
        self._equal_samples = True
        self._point_cloud_sampled = PointCloud(
            grid._sorted_points, grid._sorted_batch_ids,
            grid._batch_size)
      else:
        self._equal_samples = False
        self._point_cloud_sampled = point_cloud_sample
      self._grid = grid
      self._radii = radii
      self.max_neighbors = max_neighbors

      #Find the neighbors.
      self._samples_neigh_ranges, self._neighbors = find_neighbors(
        self._grid, self._point_cloud_sampled, self._radii, max_neighbors)

      #Original neighIds.
      aux_original_neigh_ids = tf.gather(
          self._grid._sorted_indices, self._neighbors[:, 0])
      self._original_neigh_ids = tf.concat([
        tf.reshape(aux_original_neigh_ids, [-1, 1]),
        tf.reshape(self._neighbors[:, 1], [-1, 1])], axis=-1)

      #Initialize the pdf
      self._pdf = None

  def compute_pdf(self, bandwidth, mode=0, name=None):
    """Method to compute the probability density function of a neighborhood.

    Args:
      bandwidth: float 'Tensor' of shape [D], bandwidth used to compute
        the pdf.
      mode: 'KDEMode', mode used to determine the bandwidth.
    """
    with tf.compat.v1.name_scope(
        name, "compute pdf for neighbours",
        [self, bandwidth, mode]):
      bandwidth = tf.convert_to_tensor(value=bandwidth)

      if mode == KDEMode.no_pdf:
        self._pdf = tf.ones_like(
            self._neighbors[:, 0], dtype=tf.float32)
      else:
        if self._equal_samples:
          aux_neighbors = self
        else:
          aux_neighbors = Neighborhood(self._grid, self._radii, None)
        _pdf = compute_pdf(
              aux_neighbors, bandwidth, mode.value)
        self._pdf = tf.gather(_pdf, self._neighbors[:, 0])
