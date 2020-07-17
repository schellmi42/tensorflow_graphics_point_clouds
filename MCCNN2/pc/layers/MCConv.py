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

from MCCNN2.pc.custom_ops import basis_proj
from MCCNN2.pc.layers.utils import _format_output


""" Class to represent a Monte-Carlo convolution layer

  Attributes:
    _num_features_in: An `ìnt`, the number of features per input point
    _num_features_out: An `ìnt`, the number of features to compute
    _size_hidden: An `ìnt`, the number of neurons in the hidden layer of the
      kernel MLP
    _num_dims: An `ìnt`, dimensionality of the point cloud
"""


class MCConv2Sampled:
  """ Monte-Carlo convolution layer between two point clouds.

  Args:
    num_features_in: An `int`, C_in, the number of features per input point.
    num_features_out: An `int`, C_out, the number of features to compute.
    num_dims: An `int`, dimensionality of the point cloud.
    size_hidden: An ìnt`, the number of neurons in the hidden layer of the
        kernel MLP.
    initializer_weights: A `tf.initializer` for the weights,
      default `TruncatedNormal`.
    initializer_biases: A `tf.initializer` for the biases,
      default: `zeros`.
  """

  def __init__(self,
               num_features_in,
               num_features_out,
               num_dims,
               size_hidden,
               initializer_weights=None,
               initializer_biases=None,
               name=None):
    """ Constructior, initializes weights.
    """

    with tf.compat.v1.name_scope(name, "create Monte-Carlo convolution",
                                 [self, num_features_out, num_features_in,
                                  num_features_out, size_hidden, num_dims]):
      self._num_features_in = num_features_in
      self._num_features_out = num_features_out
      self._size_hidden = size_hidden
      self._num_dims = num_dims
      if name is None:
        self._name = ''
      else:
        self._name = name

      # initialize variables
      if initializer_weights is None:
        initializer_weights = tf.initializers.TruncatedNormal
      if initializer_biases is None:
        initializer_biases = tf.initializers.zeros

      std_dev = tf.math.sqrt(1.0 / float(self._num_dims))
      hProjVecTF = tf.compat.v1.get_variable(
          self._name + '_hidden_vectors',
          shape=[self._size_hidden, self._num_dims],
          initializer=initializer_weights(stddev=std_dev),
          dtype=tf.float32,
          trainable=True)
      hProjBiasTF = tf.compat.v1.get_variable(
          self._name + '_hidden_biases',
          shape=[self._size_hidden, 1],
          initializer=initializer_biases(),
          dtype=tf.float32,
          trainable=True)
      self._basis_tf = tf.concat([hProjVecTF, hProjBiasTF], axis=1)

      std_dev = tf.math.sqrt(2.0 / \
                             float(self._size_hidden * self._num_features_in))
      self._weights = \
          tf.compat.v1.get_variable(
              self._name + '_conv_weights',
              shape=[self._size_hidden * self._num_features_in,
                     self._num_features_out],
              initializer=initializer_weights(stddev=std_dev),
              dtype=tf.float32, trainable=True)

  def _monte_carlo_convolution(self,
                               kernel_inputs,
                               neighborhood,
                               pdf,
                               features,
                               non_linearity_type=3):
    """ Method to compute a Monte-Carlo integrated convolution using a
    two layer MLP as implicit convolution kernel function.

    Args:
      kernel_inputs: A `float` `Tensor` of shape `[M, L]`, the input to the
        kernel MLP.
      neighborhood: A `Neighborhood` instance, with a `pdf` attribute.
      features: A `float` `Tensor` of shape `[N, C1]`, the input features.

    Returns:
      A `float` `Tensor` of shape ``[N,C2]`, the output features.
    """
    #Compute convolution - input to hidden layer with
    # Monte-Carlo integration - nonlinear  (RELU - 2, LRELU - 3, ELU - 4)
    weighted_features = basis_proj(kernel_inputs,
                                   neighborhood,
                                   pdf,
                                   features,
                                   self._basis_tf,
                                   non_linearity_type)
    #Compute convolution - hidden layer to output (linear)
    convolution_result = tf.matmul(
        tf.reshape(weighted_features,
                   [-1, self._num_features_in * self._size_hidden]),
        self._weights)
    return convolution_result

  def __call__(self,
               features,
               point_cloud_in: PointCloud,
               point_cloud_out: PointCloud,
               radius,
               neighborhood=None,
               bandwidth=0.2,
               return_sorted=False,
               return_padded=False,
               name=None):
    """ Computes the Monte-Carlo Convolution between two point clouds.

    Note:
      In the following, `A1` to `An` are optional batch dimensions.
      `C_in` is the number of input features.
      `C_out` is the number of output features.

    Args:
      features: A `float` `Tensor` of shape `[N_in, C_in]` or
        `[A1, ..., An,V, C_in]`.
      point_cloud_in: A 'PointCloud' instance, on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output
        features are defined.
      radius: A `float`, the convolution radius.
      neighborhood: A `Neighborhood` instance, defining the neighborhood
        with centers from `point_cloud_out` and neighbors in `point_cloud_in`.
        If `None` it is computed internally. (optional)
      bandwidth: An `int`, the bandwidth used in the kernel density
        estimation on the input point cloud. (optional)
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (optional)
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded. (optional)

    Returns:
      A `float` `Tensor` of shape
        `[N_out, C_out]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C_out]`, if `return_padded` is `True`.
    """

    with tf.compat.v1.name_scope(name, "Monte-Carlo_convolution",
                                 [features, point_cloud_in, point_cloud_out,
                                  radius, neighborhood, bandwidth,
                                  return_sorted]):
      features = tf.cast(tf.convert_to_tensor(value=features),
                         dtype=tf.float32)
      features = _flatten_features(features, point_cloud_in)
      # radius = tf.convert_to_tensor(value=radius, dtype=tf.float32)
      # bandwidth = tf.convert_to_tensor(value=bandwidth)

      #Create the radii tensor.
      radii_tensor = tf.cast(tf.repeat([radius], self._num_dims),
                             dtype=tf.float32)
      #Create the badnwidth tensor.
      bwTensor = tf.repeat(bandwidth, self._num_dims)

      if neighborhood is None:
        #Compute the grid
        grid = Grid(point_cloud_in, radii_tensor)
        #Compute the neighborhoods
        neigh = Neighborhood(grid, radii_tensor, point_cloud_out)
      else:
        neigh = neighborhood
        grid = neigh._grid
      pdf = neigh.get_pdf(bandwidth=bwTensor, mode=KDEMode.constant)

      #Compute kernel inputs.
      neigh_point_coords = tf.gather(
          grid._sorted_points, neigh._neighbors[:, 0])
      center_point_coords = tf.gather(
          point_cloud_out._points, neigh._neighbors[:, 1])
      points_diff = (neigh_point_coords - center_point_coords) / \
          tf.reshape(radii_tensor, [1, self._num_dims])
      #Compute Monte-Carlo convolution
      convolution_result = self._monte_carlo_convolution(
          points_diff, neigh, pdf, features, 3)
      return _format_output(convolution_result,
                            point_cloud_out,
                            return_sorted,
                            return_padded)

""" Class to represent a Monte-Carlo convolution layer on one point cloud.

  Attributes:
    _num_features_in: An `int`, the number of features per input point
    _num_features_out: An `int`, the number of features to compute
    _size_hidden: An `int`, the number of neurons in the hidden layer of the
      kernel MLP
    _num_dims: An `int`, dimensionality of the point cloud.
"""


class MCConv(MCConv2Sampled):
  """ Monte-Carlo convolution layer on one point cloud.

  Args:
    num_features_in: An `int` C_in, the number of features per input point.
    num_features_out: An `int` C_out, the number of features to compute.
    num_dims: An `int`, dimensionality of the point cloud.
    size_hidden: An `int`, the number of neurons in the hidden layer of the
        kernel MLP.
    initializer_weights: A `tf.initializer` for the weights,
      default `TruncatedNormal`.
    initializer_biases: A `tf.initializer` for the biases,
      default: `zeros`.
  """

  def __init__(self,
               num_features_in,
               num_features_out,
               num_dims,
               size_hidden,
               initializer_weights=None,
               initializer_biases=None,
               name=None):
    """ Constructior, initializes weights.

    """
    super(MCConv, self).__init__(num_features_in, num_features_out,
                                 size_hidden, num_dims, initializer_weights,
                                 initializer_biases, name)

  def __call__(self,
               features,
               point_cloud: PointCloud,
               radius,
               neighborhood=None,
               bandwidth=0.2,
               return_sorted=False,
               return_padded=False,
               name=None):
    """ Computes the Monte-Carlo Convolution on a point cloud.

    Note:
      In the following, `A1` to `An` are optional batch dimensions.
      `C_in` is the number of input features.
      `C_out` is the number of output features.

    Args:
      features: A `float` `Tensor` of shape `[N_in, C_in]` or
        `[A1, ..., An,V, C_in]`.
      point_cloud: A 'PointCloud' instance, on which the features are
        defined.
      radius: A `float`, the convolution radius.
      neighborhood: A `Neighborhood` instance, defining the neighborhood
        inside `point_cloud`.
        If `None` it is computed internally. (optional)
      bandwidth: An `int`, the bandwidth used in the kernel density
        estimation on the input point cloud. (optional)
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (optional)
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded. (optional)

    Returns:
      A `float` `Tensor` of shape
        `[N_out, C_out]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C_out]`, if `return_padded` is `True`.
    """
    return super(MCConv, self).__call__(features, point_cloud, point_cloud,
                                        radius, neighborhood, bandwidth,
                                        return_sorted, return_padded, name)


class MCResNet:
  """ ResNet with pre-activation using Monte-Carlo convolution layers on one
  point cloud.

  Args:
    num_features: An `int`, the number of features per input point.
    num_blocks: An `int`, the number of Resnet blocks, consisting of 2 layers
      each.
    num_dims: An `int, dimensionality of the point cloud.
    size_hidden: An `int`, the number of neurons in the hidden layer of the
        kernel MLP, can be `4, 8, 16`.
  """

  def __init__(self,
               num_features,
               num_blocks,
               num_dims,
               size_hidden,
               activation=tf.nn.relu,
               name=None):
    """ Constructior, initializes weights.

    """
    with tf.compat.v1.name_scope(
        name, "Create Monte-Carlo convolution ResNet with pre-activation",
        [num_features, num_blocks, num_dims, size_hidden, activation]):
      self._num_dims = num_dims
      self._num_blocks = num_blocks
      self._activation = activation
      self._batch_norm_layers = []
      self._conv_layers = []
      for i in range(2 * num_blocks):
        self._batch_norm_layers.append(tf.keras.layers.BatchNormalization())
        self._conv_layers.append(
            MCConv(num_features, num_features, size_hidden, num_dims))

  def __call__(self,
               features,
               point_cloud: PointCloud,
               radius,
               training,
               neighborhood=None,
               bandwidth=0.2,
               return_sorted=False,
               return_padded=False,
               name=None):
    with tf.compat.v1.name_scope(
        name,
        "Monte-Carlo convolution ResNet with pre-activation",
        [features, point_cloud, radius, training, neighborhood, bandwidth,
         return_sorted, return_padded]):
      features = tf.convert_to_tensor(value=features, dtype=tf.float32)
      features = _flatten_features(features, point_cloud)

      if neighborhood is None:

        radii_tensor = tf.repeat([radius], self._num_dims)
        #Compute the grid.
        grid = Grid(point_cloud, radii_tensor)
        #Compute the neighborhood key.
        neighborhood = Neighborhood(grid, radii_tensor)
      for i in range(self._num_blocks):
        residual = features
        features = self._batch_norm_layers[2 * i](features, training)
        features = self._activation(features)
        features = self._conv_layers[2 * i](features, point_cloud, radius,
                                            neighborhood)
        features = self._batch_norm_layers[2 * i + 1](features)
        features = self._activation(features)
        features = self._conv_layers[2 * i + 1](features, point_cloud, radius,
                                                neighborhood)
        features = features + residual
      return _format_output(features,
                            point_cloud,
                            return_sorted,
                            return_padded)
