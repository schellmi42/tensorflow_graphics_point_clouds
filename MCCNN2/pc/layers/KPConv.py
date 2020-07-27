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
"""Classes to represent point cloud convolutions"""

import tensorflow as tf
from MCCNN2.pc.utils import _flatten_features


from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc import Neighborhood
from MCCNN2.pc import KDEMode

from MCCNN2.pc.layers.utils import _format_output, kp_conv_kernel_points


def _linear_weighting(values, sigma):
  """ Linear kernel iweights for KP Conv.

  Args:
    values: A `float` `Tensor` of shape `[K, M]`, the distances to the kernel
     points.
    sigma: A `float`, the influence distance of the kernel points.

  Returns:
    A `float` `Tensor` of shape `[K, M]`.
  """
  return tf.nn.relu(1 - values / sigma)


class KPConv:
  """ A Kernel Point Convolution for 3D point clouds.

  Based on the paper [KPConv: Flexible and Deformable Convolution for Point
  Clouds. Thomas et al., 2019](https://arxiv.org/abs/1904.08889).


  Args:
    num_features_in: An `int`, C_in, the number of features per input point.
    num_features_out: An `int`, C_out, the number of features to compute.
    num_kernel_points: An ìnt`, the number of points for representing the
      kernel.
    deformable: A 'bool', indicating whether to use rigid or deformable kernel
      points.
    initializer_weights: A `tf.initializer` for the weights,
      default `TruncatedNormal`.
    initializer_biases: A `tf.initializer` for the biases,
      default: `zeros`.
  """

  def __init__(self,
               num_features_in,
               num_features_out,
               num_kernel_points=13,
               deformable=False,
               initializer_weights=None,
               name=None):
    """ Constructior, initializes variables.
    """

    with tf.compat.v1.name_scope(name, "create KP convolution",
                                 [self, num_features_out, num_features_in,
                                  num_kernel_points, deformable,
                                  initializer_weights]):
      self._num_features_in = num_features_in
      self._num_features_out = num_features_out
      self._num_kernel_points = num_kernel_points
      self._deformable = deformable
      self._num_dims = 3
      if name is None:
        self._name = ''
      else:
        self._name = name
      # initialize kernel points
      self._kernel_points = kp_conv_kernel_points(num_kernel_points,
                                                  rotate=True)
      if deformable:
        self._kernel_offsets_weights = \
            tf.compat.v1.get_variable(
                self._name + '_kernel_point_offset_weights',
                shape=[self._num_features_in,
                       self._num_kernel_points * 3],
                initializer=tf.initializers.zeros,
                dtype=tf.float32,
                trainable=True)
        self._get_offsets = self._kernel_offsets
      else:
        def _zero(*args, **kwargs):
          """ Replaces `_kernel_offsets` with zeros for rigid KPConv.
          """
          return tf.constant(0.0, dtype=tf.float32)
        self._get_offsets = _zero

      # initialize variables
      if initializer_weights is None:
        initializer_weights = tf.initializers.TruncatedNormal

      std_dev = tf.math.sqrt(2.0 / \
                             float(self._num_features_in))
      self._weights = \
          tf.compat.v1.get_variable(
              self._name + '_conv_weights',
              shape=[self._num_kernel_points,
                     self._num_features_in,
                     self._num_features_out],
              initializer=initializer_weights(stddev=std_dev),
              dtype=tf.float32,
              trainable=True)

  def _kp_conv(self,
               kernel_input,
               neighborhood,
               features):
    """ Method to compute a kernel point convolution using linear interpolation
    of the kernel weights.

    Note: In the following
      `D` is the dimensionality of the points cloud (=3)
      `M` is the number of neighbor pairs
      'C1`is the number of input features
      `C2` is the number of output features
      `N1' is the number of input points
      `N2' is the number of ouput points

    Args:
      kernel_inputs: A `float` `Tensor` of shape `[M, D]`, the input to the
        kernel, i.e. the distances between neighbor pairs.
      neighborhood: A `Neighborhood` instance.
      features: A `float` `Tensor` of shape `[N1, C1]`, the input features.

    Returns:
      A `float` `Tensor` of shape `[N2, C2]`, the output features.
    """
    # neighbor pairs ids
    neighbors = neighborhood._original_neigh_ids
    # kernel weights from distances, shape [K, M]
    kernel_offsets = self._get_offsets(kernel_input, neighborhood, features)
    points_diff = tf.expand_dims(kernel_input, 0) - \
        (tf.expand_dims(self._kernel_points, 1) + kernel_offsets)
    points_dist = tf.linalg.norm(points_diff, axis=2)
    kernel_weights = _linear_weighting(points_dist, self._sigma)

    # weighted features per kernel and input features dim, shape [K, M, C1]
    features_per_nb = tf.gather(features, neighbors[:, 0])
    weighted_features = tf.expand_dims(features_per_nb, axis=0) * \
        tf.expand_dims(kernel_weights, axis=2)

    # matrix multiplication treating kernel dimension as batch dimension
    # shape [K, M, C1] x [K, C1, C2] -> [K, M, C2]
    convolution_result = tf.matmul(weighted_features, self._weights)
    # sum over kernel dimension, shape [M, C2]
    convolution_result = tf.reduce_sum(convolution_result, axis=0)
    # sum over neighbors, shape [N2, C2]
    return  tf.math.unsorted_segment_sum(convolution_result,
                                         neighbors[:, 1],
                                         self._num_output_points)

  def __call__(self,
               features,
               point_cloud_in: PointCloud,
               point_cloud_out: PointCloud,
               conv_radius,
               neighborhood=None,
               kernel_influence_dist=None,
               return_sorted=False,
               return_padded=False,
               name=None):
    """ Computes the Kernel Point Convolution between two point clouds.

    Note:
      In the following, `A1` to `An` are optional batch dimensions.
      `C_in` is the number of input features.
      `C_out` is the number of output features.

    Args:
      features: A `float` `Tensor` of shape `[N1, C_in]` or
        `[A1, ..., An,V, C_in]`.
      point_cloud_in: A 'PointCloud' instance, on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output
        features are defined.
      conv_radius: A `float`, the convolution radius.
      neighborhood: A `Neighborhood` instance, defining the neighborhood
        with centers from `point_cloud_out` and neighbors in `point_cloud_in`.
        If `None` it is computed internally. (optional)
      kernel_influence_dist = A `float`, the influence distance of the kernel
        points. If `None` uses `conv_radius / 2.5`, as suggested in Section 3.3
        of the paper. (optional)
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (optional)
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded. (optional)

    Returns:
      A `float` `Tensor` of shape
        `[N2, C_out]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C_out]`, if `return_padded` is `True`.
    """

    with tf.compat.v1.name_scope(name, "Kernel Point_convolution",
                                 [features, point_cloud_in, point_cloud_out,
                                  conv_radius, neighborhood,
                                  kernel_influence_dist, return_sorted,
                                  return_padded]):
      features = tf.cast(tf.convert_to_tensor(value=features),
                         dtype=tf.float32)
      features = _flatten_features(features, point_cloud_in)
      self._num_output_points = point_cloud_out._points.shape[0]

      if kernel_influence_dist is None:
        self._sigma = conv_radius / 2.5
      else:
        self._sigma = tf.convert_to_tensor(
          value=kernel_influence_dist, dtype=tf.float32)

      #Create the radii tensor.
      radii_tensor = tf.cast(tf.repeat([conv_radius], self._num_dims),
                             dtype=tf.float32)

      if neighborhood is None:
        #Compute the grid
        grid = Grid(point_cloud_in, radii_tensor)
        #Compute the neighborhoods
        neigh = Neighborhood(grid, radii_tensor, point_cloud_out)
      else:
        neigh = neighborhood

      #Compute kernel inputs.
      neigh_point_coords = tf.gather(
          point_cloud_in._points, neigh._original_neigh_ids[:, 0])
      center_point_coords = tf.gather(
          point_cloud_out._points, neigh._original_neigh_ids[:, 1])
      points_diff = (neigh_point_coords - center_point_coords) / \
          tf.reshape(radii_tensor, [1, self._num_dims])
      #Compute Monte-Carlo convolution
      convolution_result = self._kp_conv(points_diff, neigh, features)
      return _format_output(convolution_result,
                            point_cloud_out,
                            return_sorted,
                            return_padded)

  def _kernel_offsets(self,
                      kernel_input,
                      neighborhood,
                      features):
    """ Method to compute the kernel offsets for deformable KPConv
    using a rigid KPConv.

    As described in Section 3.2 of [KPConv: Flexible and Deformable Convolution
    for Point Clouds. Thomas et al., 2019](https://arxiv.org/abs/1904.08889).

    Note: In the following
      `D` is the dimensionality of the point cloud (=3)
      `M` is the number of neighbor pairs
      'C1`is the number of input features
      `N1' is the number of input points
      `N2' is the number of ouput points
      `K` is the number of kernel points

    Args:
      kernel_inputs: A `float` `Tensor` of shape `[M, D]`, the input to the
        kernel, i.e. the distances between neighbor pairs.
      neighborhood: A `Neighborhood` instance.
      features: A `float` `Tensor` of shape `[N1, C1]`, the input features.

    Returns:
      A `float` `Tensor` of shape `[K, M, 3]`, the offsets.
    """
    # neighbor pairs ids
    neighbors = neighborhood._original_neigh_ids
    # kernel weights from distances, shape [K, M]
    points_diff = tf.expand_dims(kernel_input, 0) - \
        tf.expand_dims(self._kernel_points, 1)
    points_dist = tf.linalg.norm(points_diff, axis=2)
    kernel_weights = _linear_weighting(points_dist, self._sigma)

    # weighted features per kernel and input features dim, shape [K, M, C1]
    features_per_nb = tf.gather(features, neighbors[:, 0])
    weighted_features = tf.expand_dims(features_per_nb, axis=0) * \
        tf.expand_dims(kernel_weights, axis=2)

    # matrix multiplication treating kernel dimension as batch dimension
    # shape [K, M, C1] x [K, C1, D] -> [K, M, 3*K]
    convolution_result = tf.matmul(weighted_features,
                                   self._kernel_offsets_weights)
    # sum over kernel dimension, shape [M, 3*K]
    convolution_result = tf.reduce_sum(convolution_result, axis=0)
    # sum over neighbors, shape [N2, 3*K]
    offset_per_center = tf.math.unsorted_segment_sum(convolution_result,
                                                     neighbors[:, 1],
                                                     self._num_output_points)
    # project back onto neighbor pairs, shape [M, 3*K]
    offset_per_nb = tf.gather(offset_per_center, neighbors[:, 1])
    # reshape to shape [K, M, 3]
    return  tf.transpose(tf.reshape(offset_per_nb,
                                    [neighbors.shape[0],
                                     self._num_kernel_points,
                                     3]),
                         [1, 0, 2])
