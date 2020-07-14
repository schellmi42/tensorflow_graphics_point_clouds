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
