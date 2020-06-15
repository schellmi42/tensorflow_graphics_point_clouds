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
"""Class to represent axis aligned bounding box of point clouds"""

import tensorflow as tf


class AABB:
  """Class to represent axis aligned bounding box of point clouds.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Attributes:
    aabbMin_ (float tensor bxd): List of minimum points of the bounding boxes.
    aabbMax_ (float tensor bxd): List of maximum points of the bounding boxes.
    batchSize_ (int): Size of the batch.
    batchShape_: An int tensor of shape [A1,...,An]
  """

  def __init__(self, point_cloud, name=None):
    """Constructor.

    Args:
      Pointcloud (PointCloud):  Point cloud from which to compute the
        bounding box.
    """
    with tf.compat.v1.name_scope(
        name, "bounding box constructor", [self, point_cloud]):
      self.batchSize_ = point_cloud.batchSize_
      self.batchShape_ = point_cloud.batchShape_
      self.aabbMin_ = tf.math.unsorted_segment_min(
          data=point_cloud.pts_, segment_ids=point_cloud.batchIds_,
          num_segments=self.batchSize_) - 1e-9
      self.aabbMax_ = tf.math.unsorted_segment_max(
          data=point_cloud.pts_, segment_ids=point_cloud.batchIds_,
          num_segments=self.batchSize_) + 1e-9

  def get_diameter(self, ord='euclidean', name=None):
    """ Returns the diameter of the bounding box.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      ord:    Order of the norm. Supported values are `'euclidean'`,
          `1`, `2`, `np.inf` and any positive real number yielding the
          corresponding p-norm. Default is `'euclidean'`.
    Return:
      diam:   tensor [A1,..An] diameters of the bound boxes
    """

    with tf.compat.v1.name_scope(
        name, "Compute diameter of bounding box",
        [self, ord]):
      diam = tf.linalg.norm(self.aabbMax_ - self.aabbMin_, ord=ord, axis=-1)
      if self.batchShape_ is None:
        return diam
      else:
        return tf.reshape(diam, self.batchShape_)
