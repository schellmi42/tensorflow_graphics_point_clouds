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
  In the following `D` is the spatial dimensionality of the points,
  `N` is the number of (samples) points, and `M` is the total number of
  adjacencies.

Attributes:
  _point_cloud_sampled: 'PointCloud', samples point cloud.
  _grid : 'Grid', regular grid data structure.
  _radii: `float` `Tensor` of shape [D], radius used to select the neighbors.
  _samples_neigh_ranges: `int` `Tensor` of shape `[N]`, end of the ranges per
    sample.
  _neighbors: `int` `Tensor` of shape `[M,2]`, indices of the neighbor point,
    with respect to the sorted point in the grid, and the sample for each
    neighbor.
  _original_neigh_ids: `int` `Tensor` of shape `[M,2]`, indices of the
    neighbor point, with respect to the points in the input point cloud,
    and the sample for each neighbor.
  _pdf: `float` `Tensor` of shape `[M]`, PDF value for each neighbor.
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
    radius: A `float` `Tensor` of shape `[D]`, the radius used to select the
      neighbors.
    point_cloud_sample: A 'PointCloud' instance. Samples point cloud.
      If None, the sorted points from the grid will be used.
    max_neighbors: An `int`, maximum number of neighbors per sample,
      if `0` all neighbors are selected.
  """

  def __init__(self,
               grid: Grid,
               radius,
               point_cloud_sample=None,
               max_neighbors=0,
               name=None):
    with tf.compat.v1.name_scope(
        name, "constructor for neighbourhoods of point clouds",
        [self, grid, radius, point_cloud_sample, max_neighbors]):
      radii = tf.reshape(tf.cast(tf.convert_to_tensor(value=radius),
                                 tf.float32), [-1])
      if radii.shape[0] == 1:
        radii = tf.repeat(radius, grid._point_cloud._dimension)
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

      self._transposed = None

  def _compute_pdf(self, bandwidth=0.2, mode=0, name=None):
    """Method to compute the probability density function of the neighborhoods.

    Args:
      bandwidth: float ``Tensor`` of shape `[D]`, bandwidth used to compute
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

  def get_pdf(self, **kwargs):
    """ Method which returns the pdfs of the neighborhoods.
    If no pdf was computed before, it will compute one using the provided
    arguments.

    Args:
      **kwargs: if no pdf is available, these arguments will be passed to
      `_compute_pdf`.(optional)

    Returns:
      A `float` `Tensor` of shape `[M]`, the estimated densities.
    """
    if self._pdf is None:
      self._compute_pdf(**kwargs)
    return self._pdf
  
  def get_grid(self):
    """ Returns the grid used for neighborhood computation.
    """
    return self._grid

  def transpose(self):
    """ Returns the transposed neighborhood where center and neighbor points
    are switched. (faster than recomputing)
    """
    if self._transposed is None:
      if self._equal_samples:
        self._transposed = self
      else:
        grid = Grid(self._point_cloud_sampled, self._radii)
        self._transposed = Neighborhood(
            grid, self._radii, self._grid._point_cloud)
    return self._transposed
    # return _NeighborhoodTransposed(self)


class _NeighborhoodTransposed(Neighborhood):
  """ Builds a transposed neighborhood where center and neighbor points
    are switched. (in some cases slower than recomputing)
  """

  def __init__(self, neighborhood):
    if neighborhood._equal_samples:
      self = neighborhood
    else:
      self._equal_samples = False
      self._radii = neighborhood._radii
      self._point_cloud_sampled = neighborhood._grid._point_cloud
      self._grid = Grid(neighborhood._point_cloud_sampled, neighborhood._radii)

      self._original_neigh_ids = tf.reverse(
          neighborhood._original_neigh_ids, axis=[1])
      sort_centers = tf.argsort(self._original_neigh_ids[:, 1])
      self._original_neigh_ids = tf.gather(
          self._original_neigh_ids, sort_centers)

      unsorted_indices = tf.math.invert_permutation(self._grid._sorted_indices)
      aux_neigh_ids = tf.gather(
          unsorted_indices, self._original_neigh_ids[:, 0])
      self._neighbors = tf.concat([
        tf.reshape(aux_neigh_ids, [-1, 1]),
        tf.reshape(self._original_neigh_ids[:, 1], [-1, 1])], axis=-1)

      num_neighbors = neighborhood._original_neigh_ids.shape[0]
      num_center_points = self._point_cloud_sampled._points.shape[0]
      self._samples_neigh_ranges = tf.math.unsorted_segment_max(
          tf.range(1, num_neighbors + 1),
          self._neighbors[:, 1],
          num_center_points)

      self._pdf = None
