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
"""Utility methods for point cloud layers."""

import tensorflow as tf
import numpy as np

tf_pi = tf.convert_to_tensor(np.pi)


def _format_output(features, point_cloud, return_sorted, return_padded):
  """ Method to format and sort the output of a point cloud convolution layer.

  Note:
    In the following, `A1` to `An` are optional batch dimensions.

  Args:
    features: A `float` `Tensor`  of shape `[N, C]`.
    point_cloud: A `PointCloud` instance, on which the `feautres` are defined.
    return_sorted: A `bool`, if 'True' the output tensor is sorted
      according to the sorted batch ids of `point_cloud`.
    return_padded: A `bool`, if 'True' the output tensor is sorted and
      zero padded.

  Returns:
    A `float` `Tensor` of shape
      `[N, C]`, if `return_padded` is `False`
    or
      `[A1, ..., An, V, C]`, if `return_padded` is `True`.
  """

  if return_padded:
    unflatten = point_cloud.get_unflatten()
    features = unflatten(features)
  elif return_sorted:
    features = tf.gather(features, point_cloud._sorted_indices_batch)
  return features


def random_rotation(points, name=None):
  """ Method to rotate 3D points randomly.

  Args:
    points: A `float` `Tensor` of rshape `[N, 3]`.

  Returns:
    A `float` `Tensor` of the same shape as `points`.

  """
  with tf.compat.v1.name_scope(name, 'random point rotation', [points]):
    points = tf.convert_to_tensor(value=points)
    angles = tf.random.uniform([3], 0, 2 * np.pi)
    sine = tf.math.sin(angles)
    cosine = tf.math.cos(angles)
    Rx = tf.stack(([1.0, 0.0, 0.0],
                   [0.0, cosine[0], -sine[0]],
                   [0.0, sine[0], cosine[0]]), axis=1)
    Ry = tf.stack(([cosine[1], 0, sine[1]],
                   [0.0, 1.0, 0.0],
                   [-sine[1], 0.0, cosine[1]]), axis=1)
    Rz = tf.stack(([cosine[2], -sine[2], 0.0],
                   [sine[2], cosine[2], 0.0],
                   [0.0, 0.0, 1.0]), axis=1)
    R = tf.matmul(tf.matmul(Rx, Ry), Rz)
    return tf.matmul(points, R)


def kp_conv_kernel_points(num_points, rotate=True, name=None):
  """ Conputes the initial positions of the kernel points for KPConv.

  The points are located at positions as described in Appendix B of
  [KPConv: Flexible and Deformable Convolution for Point Clouds. Thomas et
  al.,
  2019](https://arxiv.org/abs/1904.08889).

  Args:
    num_points: An `int`, the number of kernel points, must be in [5, 7, 13].
    rotate: A 'bool', if `True` a random rotation is applied to the points.

  Returns:
    A `float` `Tensor` of shape `[num_points, 3]`.
  """
  with tf.compat.v1.name_scope(
      name, "KPConv kernel points",
      [num_points, rotate]):
    print()
    if num_points not in [5, 7, 13]:
      raise ValueError('KPConv currently only supports kernel sizes' + \
                       ' [5, 7, 13]')
    if num_points == 5:
      # Tetrahedron
      points = tf.Variable([[0, 0, 0],
                            [0, 0, 1],
                            [tf.sqrt(8 / 9), 0, -1 / 3],
                            [- tf.sqrt(2 / 9), tf.sqrt(2 / 3), - 1 / 3],
                            [-tf.sqrt(2 / 9), - tf.sqrt(2 / 3), -1 / 3]],
                           dtype=tf.float32)
    elif num_points == 7:
      # Octahedron
      points = tf.Variable([[0, 0, 0],
                            [1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1, 0],
                            [0, 0, 1],
                            [0, 0, -1]], dtype=tf.float32)
    elif num_points == 13:
      # Icosahedron
      phi = (1 + tf.sqrt(5)) / 2
      points = tf.Variable([[0, 0, 0],
                            [0, 1, phi],
                            [0, 1, -phi],
                            [0, -1, phi],
                            [0, -1, -phi],
                            [1, phi, 0],
                            [1, -phi, 0],
                            [-1, phi, 0],
                            [-1, -phi, 0],
                            [phi, 0, 1],
                            [-phi, 0, 1],
                            [phi, 0, -1],
                            [-phi, 0, -1]], dtype=tf.float32)
    if rotate:
      points = random_rotation(points)
    return points


def _identity(features, *args, **kwargs):
  """ Simple identity layer, to be used as placeholder.

  Used to replace projection shortcuts, if not desired.
  """
  return features


def positional_encoding(values, order, include_original=False, name=None):
  with tf.compat.v1.name_scope(
      name, "positional encoding", [values, order, include_original]):
    values = tf.convert_to_tensor(value=values, dtype=tf.float32)
    num_dims = values.shape.ndims
    frequencies = tf_pi * tf.pow(2, tf.range(0, order, dtype=tf.float32))
    broadcast_shape = tf.concat((tf.repeat([1], num_dims - 1), [-1, 1]),
                                axis=0)
    # input to trigonometry encoding, shape [...,  L, D]
    modulated_values = tf.expand_dims(values, -2) *\
        tf.reshape(frequencies, broadcast_shape)
    # encoding, shape [..., L, 2, D]
    encoding = tf.stack((tf.sin(modulated_values), tf.cos(modulated_values)),
                        axis=-2)
    output_shape = tf.concat((values.shape[:-1], [values.shape[-1] * 2 * order]),
                             axis=0)
    encoding = tf.reshape(encoding, output_shape)
    if include_original:
      encoding = tf.concat((values, encoding), axis=-1)
    return encoding

