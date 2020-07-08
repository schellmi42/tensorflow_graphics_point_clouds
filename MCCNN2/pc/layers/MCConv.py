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
"""Class to represent point cloud convolution"""

import tensorflow as tf
from MCCNN2.pc.utils import _flatten_features


from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc import Neighborhood
from MCCNN2.pc import KDEMode

from MCCNN2.pc.custom_ops import basis_proj


class MCConv2Sampled:
  """ Class to represent a Monte-Carlo convolution layer

    Attributes:
      _num_features_in: An `ìnt`, the number of features per input point
      _num_features_out: An `ìnt`, the number of features to compute
      _size_hidden: An `ìnt`, the number of neurons in the hidden layer of the
        kernel MLP
      _num_dims: An `ìnt`, dimensionality of the point cloud
    conv_name: A `string`, name for the operation
  """

  def __init__(self,
               num_features_in,
               num_features_out,
               size_hidden,
               num_dims,
               initializer_weights=None,
               initializer_biases=None,
               conv_name=None):
    """ Constructior, initializes weights

    Args:
    num_features_in: An `int` C_in, the number of features per input point
    num_features_out: An `int` C_out, the number of features to compute
    size_hidden: An ìnt`, the number of neurons in the hidden layer of the
        kernel MLP
    num_dims: An `int`, dimensionality of the point cloud
    initializer_weights: A `tf.initializer` for the weights,
      default `TruncatedNormal`
    initializer_biases: A `tf.initializer` for the biases,
      default: `zeros`
    conv_name: A `string`, name for the operation
    """

    with tf.compat.v1.name_scope(conv_name, "create Monte-Carlo convolution",
                                 [self, num_features_out, num_features_in,
                                  num_features_out, size_hidden, num_dims]):
      self._num_features_in = num_features_in
      self._num_features_out = num_features_out
      self._size_hidden = size_hidden
      self._num_dims = num_dims
      if conv_name is None:
        self._conv_name = ''
      else:
        self._conv_name = conv_name

      # initialize variables
      if initializer_weights is None:
        initializer_weights = tf.initializers.TruncatedNormal
      if initializer_biases is None:
        initializer_biases = tf.initializers.zeros

      std_dev = tf.math.sqrt(1.0 / float(self._num_dims))
      hProjVecTF = tf.compat.v1.get_variable(
          self._conv_name + '_hidden_vectors',
          shape=[self._size_hidden, self._num_dims],
          initializer=initializer_weights(stddev=std_dev),
          dtype=tf.float32,
          trainable=True)
      hProjBiasTF = tf.compat.v1.get_variable(
          self._conv_name + '_hidden_biases',
          shape=[self._size_hidden, 1],
          initializer=initializer_biases(),
          dtype=tf.float32,
          trainable=True)
      self._basis_tf = tf.concat([hProjVecTF, hProjBiasTF], axis=1)

      std_dev = tf.math.sqrt(2.0 / \
                             float(self._size_hidden * self._num_features_in))
      self._weights = \
          tf.compat.v1.get_variable(
              self._conv_name + '_conv_weights',
              shape=[self._size_hidden * \
                     self._num_features_in,
                     self._num_features_out],
              initializer=initializer_weights(stddev=std_dev),
              dtype=tf.float32, trainable=True)

  def __call__(self,
               features,
               point_cloud_in: PointCloud,
               point_cloud_out: PointCloud,
               radius,
               neighborhood=None,
               bandwidth=0.2,
               return_sorted=False,
               name=None):
    """ Computes the Monte-Carlo Convolution

    Note:
      In the following, A1 to An are optional batch dimensions.
      C_in is the number of input features.
      C_out is the number of output features.

    Args:
      features: A `float` Tensor of shape [N_in, C_in] or
        [A1, ..., An,V, C_in],
        the size must be the same as the points in the input point cloud.
      point_cloud_in: A 'PointCloud' instance, represents the input
        point cloud.
      point_cloud_out: A `PointCloud` instance, represents the output
        point cloud.
      radius: A `float`, the convolution radius.
      neighborhood: A `Neighborhood` instance, defining the neighborhood
        with centers from `point_cloud_out` and neighbors in `point_cloud_in`.
        If `None` it is computed internally. (optional)
      bandwidth: An `int`, the bandwidth used in the kernel density
        estimation on the input point cloud.
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (default: False)

      Returns:
        Tensor with shape [N_out, C_out]
    """

    with tf.compat.v1.name_scope(name, "Monte-Carlo_convolution",
                                 [features, point_cloud_in, point_cloud_out,
                                  radius, neighborhood, bandwidth,
                                  return_sorted]):
      features = tf.convert_to_tensor(value=features, dtype=tf.float32)
      features = _flatten_features(features, point_cloud_in)
      radius = tf.convert_to_tensor(value=radius, dtype=tf.float32)
      bandwidth = tf.convert_to_tensor(value=bandwidth)

      #Create the radii tensor.
      radii_tensor = tf.repeat([radius], self._num_dims)
      #Create the badnwidth tensor.
      bwTensor = tf.repeat(bandwidth, self._num_dims)

      if neighborhood is None:
        #Compute the grid.
        grid = Grid(point_cloud_in, radii_tensor)

        #Compute the neighborhood key.
        neigh = Neighborhood(grid, radii_tensor, point_cloud_out)
      else:
        neigh = neighborhood
        grid = neigh._grid
      neigh.compute_pdf(bwTensor, mode=KDEMode.constant)

      #Compute kernel inputs.
      neigh_point_coords = tf.gather(
          grid._sorted_points, neigh._neighbors[:, 0])
      center_point_coords = tf.gather(
          point_cloud_out._points, neigh._neighbors[:, 1])
      points_diff = (neigh_point_coords - center_point_coords) / \
          tf.reshape(radii_tensor, [1, self._num_dims])

      #Compute convolution (RELU - 2, LRELU - 3, ELU - 4)
      weighted_features = basis_proj(
          points_diff, neigh, features, self._basis_tf, 3)

      #Compute the convolution.
      convolution_result = tf.matmul(tf.reshape(
          weighted_features, [-1, self._num_features_in * self._size_hidden]),
          self._weights)
      if return_sorted:
        convolution_result = tf.gather(convolution_result,
                                       point_cloud_out.sortedIndicesBatch_)
      return convolution_result


class MCConv(MCConv2Sampled):
  """ Class to represent a Monte-Carlo convolution layer on one point cloud

    Attributes:
      _num_features_in: Integer, the number of features per input point
      _num_features_out: Integer, the number of features to compute
      _size_hidden: Integer, the number of neurons in the hidden layer of the
        kernel MLP
      _num_dims: Integer, dimensionality of the point cloud
      _conv_name: String, name for the operation
  """

  def __init__(self,
               num_features_in,
               num_features_out,
               size_hidden,
               num_dims,
               conv_name=None):
    """ Constructior, initializes weights

    Args:
    num_features_in: Integer C_in, the number of features per input point
    num_features_out: Integer C_out, the number of features to compute
    size_hidden: Integer, the number of neurons in the hidden layer of the
        kernel MLP
    num_dims: Integer, dimensionality of the point cloud
    conv_name: String, name for the operation
    """
    super(MCConv, self).__init__(num_features_in, num_features_out,
                                 size_hidden, num_dims, None, None, conv_name)

  def __call__(self,
               features,
               point_cloud: PointCloud,
               radius,
               neighborhood=None,
               bandwidth=0.2,
               return_sorted=False,
               name=None):
    """ Computes the Monte-Carlo Convolution

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      features: A float `Tensor` of shape [N, C_in] or [A1, ..., An, V, C_in],
        the size must be the same as the points in the input point cloud.
      point_cloud: A `PointCloud` instance
      radius: A `float`, the convolution radius.
      neighborhood: A `Neighborhood` instance, defining the neighborhood
        inside `point_cloud`.
        If `None` it is computed internally. (optional)
      bandwidth: An `int`, the bandwidth used in the kernel density
        estimation on the input point cloud.
      return_sorted: A boolean, if 'True' the output tensor is sorted
        according to the batch_ids.

      Returns:
        `Tensor` with shape [N,C_out]
    """
    return super(MCConv, self).__call__(features, point_cloud, point_cloud,
                                radius, neighborhood, bandwidth, return_sorted,
                                name)
