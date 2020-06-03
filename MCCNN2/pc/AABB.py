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

    Attributes:
        aabbMin_ (float tensor bxd): List of minimum points of the bounding boxes.
        aabbMax_ (float tensor bxd): List of maximum points of the bounding boxes.
        batch_size_ (int): Size of the batch.
    """

    def __init__(self, Pointcloud, name=None):
        """Constructor.

        Args:
            Pointcloud (Pointcloud): Point cloud from which to compute the bounding box.
        """
        with tf.compat.v1.name_scope(name, "bounding box constructor", [self, Pointcloud]):
            self.batch_size_ = Pointcloud.batch_size_
            self.aabbMin_ = tf.math.unsorted_segment_min(Pointcloud.pts_, Pointcloud.batchIds_, self.batch_size_)-1e-9
            self.aabbMax_ = tf.math.unsorted_segment_max(Pointcloud.pts_, Pointcloud.batchIds_, self.batch_size_)+1e-9
            