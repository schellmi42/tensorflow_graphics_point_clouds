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
"""Class to represent monte-carlo point cloud convolution"""

import tensorflow as tf
from MCCNN2.pc.utils import _flatten_features
from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc import Neighborhood


class GlobalMaxPooling:

  def __call__(self, features, point_cloud: PointCloud, name=None):
    """ Performs a global max pooling on a point cloud
    Args:
      features: A float `Tensor` of shape [N,D] or [A1,...,An,V,D].
      point_cloud: A PointCloud instance.
    Returns:
      A float `Tensor` of shape [batch_size,D]
    """
    with tf.compat.v1.name_scope(
        name, "global max pooling ", [features, point_cloud]):
      features = _flatten_features(features, point_cloud)
      return tf.math.unsorted_segment_max(
          features,
          segment_ids=point_cloud._batch_ids,
          num_segments=point_cloud._batch_size)


class GlobalAveragePooling:

  def __call__(self, features, point_cloud: PointCloud, name=None):
    """ performs a global average pooling on a point cloud
    Args:
      features: A float `Tensor` of shape [N,D] or [A1,...,An,V,D].
      point_cloud: A PointCloud instance.
    Returns:
      A float `Tensor` of shape [batch_size,D].
    """
    with tf.compat.v1.name_scope(
        name, "global average pooling ", [features, point_cloud]):
      features = _flatten_features(features, point_cloud)
      return tf.math.unsorted_segment_mean(
          features,
          segment_ids=point_cloud._batch_ids,
          num_segments=point_cloud._batch_size)


class _LocalPointPooling:

  def __call__(self, pool_op, features, point_cloud_in: PointCloud,
               point_cloud_out: PointCloud, pooling_radius,
               return_sorted=False, name=None, default_name="custom pooling"):
    """ Computes a local pooling between two point clouds specified by pool_op
    Args:
      pool_op: A function of type `tf.math.unsorted_segmented_*`
      features: A float `Tensor` of shape [N_in,D] or [A1,...,An,V,D].
      point_cloud_in: A `PointCloud` instance on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output features
        are defined.
      pooling_radius: A float or a float `Tensor` of shape [D]
      return_sorted: A boolean, if 'True' the output tensor is sorted
        according to the batch_ids.
    Returns:
      A float `Tensor` of shape [N_out,D].
    """
    with tf.compat.v1.name_scope(
        name, default_name,
        [features, point_cloud_in, point_cloud_out, return_sorted,
         pooling_radius]):
      features = _flatten_features(features, point_cloud_in)
      pooling_radius = tf.convert_to_tensor(
          value=pooling_radius, dtype=tf.float32)
      if pooling_radius.shape[0] == 1:
        pooling_radius = tf.repeat(pooling_radius, point_cloud_in.dimension_)

      # Compute the AABB.
      aabb_in = point_cloud_in.get_AABB()

      # Compute the grid.
      grid_in = Grid(point_cloud_in, aabb_in, pooling_radius)

      # Compute the neighborhood keys.
      neigh = Neighborhood(grid_in, pooling_radius, point_cloud_out)
      # -----------------
      # quick fix for 2D input
      # mask points with different batch_id
      features_on_neighbors = tf.gather(
          features, neigh._original_neigh_ids[:, 0])
      batch_ids_in = tf.gather(
          point_cloud_in._batch_ids, neigh._original_neigh_ids[:, 0])
      batch_ids_out = tf.gather(
          point_cloud_out._batch_ids, neigh._original_neigh_ids[:, 1])
      batch_mask = batch_ids_in == batch_ids_out
      features_on_neighbors = tf.boolean_mask(
          features_on_neighbors, batch_mask)
      neigh_out = tf.boolean_mask(neigh._original_neigh_ids[:, 1], batch_mask)
      # -------------

      # Pool the features in the neighborhoods
      features_out = pool_op(
          data=features_on_neighbors,
          segment_ids=neigh_out,
          num_segments=point_cloud_out._points.shape[0])
      if return_sorted:
        features_out = tf.gather(
            features_out, point_cloud_out._sorted_indices_batch)
      return features_out


class MaxPooling(_LocalPointPooling):

  def __call__(self, features, point_cloud_in: PointCloud,
               point_cloud_out: PointCloud, pooling_radius,
               return_sorted=False, name=None):
    """ Computes a local max pooling between two point clouds
    Args:
      features: A float `Tensor` of shape [N_in,D] or [A1,...,An,V,D].
      point_cloud_in: A PointCloud instance on which the features are defined.
      point_cloud_out: A PointCloud instance, on which the output features
        are defined.
      pooling_radius: A float or a float `Tensor` of shape [D]
      return_sorted: A boolean, if 'True' the output tensor is sorted
        according to the batch_ids.
    Returns:
      A float `Tensor` of shape [N_out,D].
    """
    return super(MaxPooling, self).__call__(
        tf.math.unsorted_segment_max,
        features, point_cloud_in, point_cloud_out, pooling_radius,
        return_sorted, name, default_name="max pooling")


class AveragePooling(_LocalPointPooling):

  def __call__(self, features, point_cloud_in: PointCloud,
               point_cloud_out: PointCloud, pooling_radius,
               return_sorted=False, name=None):
    """ Computes a local average pooling between two point clouds
    Args:
      features: A float `Tensor` of shape [N_in,D] or [A1,...,An,V,D].
      point_cloud_in: A PointCloud instance on which the features are defined.
      point_cloud_out: A PointCloud instance, on which the output features
        are defined.
      pooling_radius: A float or a float `Tensor` of shape [D]
      return_sorted: A boolean, if 'True' the output tensor is sorted
        according to the batch_ids.
    Returns:
      A float `Tensor` of shape [N_out,D].
    """
    return super(AveragePooling, self).__call__(
        tf.math.unsorted_segment_mean,
        features, point_cloud_in, point_cloud_out, pooling_radius,
        return_sorted, name, default_name="average pooling")
