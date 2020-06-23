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


class GlobalMaxPooling:

  def __call__(self, features, point_cloud: PointCloud, name=None):
    with tf.compat.v1.name_scope(
        name, "global max pooling ", [features, point_cloud]):
      features = _flatten_features(features, point_cloud)
      return tf.math.unsorted_segment_max(features,
                                          segment_ids=point_cloud.batchIds_,
                                          num_segments=point_cloud.batchSize_)


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
      return tf.math.unsorted_segment_max(features,
                                          segment_ids=point_cloud.batchIds_,
                                          num_segments=point_cloud.batchSize_)


class MaxPooling:

  def __call__(self, features, point_cloud_in: PointCloud,
              point_cloud_out: PointCloud, name=None):
    """ performs a global average pooling on a point cloud
    Args:
      features: A float `Tensor` of shape [N,D] or [A1,...,An,V,D].
      point_cloud: A PointCloud instance.
    Returns:
      A float `Tensor` of shape [batch_size,D].
    """
    with tf.compat.v1.name_scope(
        name, "average pooling ", [features, point_cloud_in, point_cloud_out]):
      features = _flatten_features(features, point_cloud_in)
