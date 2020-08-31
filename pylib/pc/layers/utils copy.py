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

def positional_encoding(values, order, include_original=False, name=None):
  """ Positional Encoding as described in NERF paper.

  Args:
    values: A `float` `Tensor` of shape `[A1, ..., An, D]`.
    order: An `int`, the order `L` of the positional encoding.
    include_original: A `bool`, if `True` then `values` is appended
      to the encoding.

    Returns:
      A `float` `Tensor` of shape
        `[A1, ..., An, D*L*2]`,
      or
        `[A1, ..., An, D*(L*2+1)]`, if `include_original` is `True`.
  """
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
    output_shape = tf.concat((values.shape[:-1],
                              [values.shape[-1] * 2 * order]),
                             axis=0)
    encoding = tf.reshape(encoding, output_shape)
    if include_original:
      encoding = tf.concat((values, encoding), axis=-1)
    return encoding
