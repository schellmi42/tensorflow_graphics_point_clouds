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
"""Class to represent point clouds"""

import tensorflow as tf

class PointCloud:
    """Class to represent a point cloud.

    Attributes:
        pts_ (float tensor nxd): List of points.
        batchIds_ (int tensor n): List of batch ids associated with the points.
        batchSize_ (int): Size of the batch.
    """

    def __init__(self, pPts, pBatchIds, pBatchSize, name=None):
        """Constructor.

        Args:
            pPts (float tensor nxd): List of points.
            pBatchIds (int tensor n): List of batch ids associated with the points.
            pBatchSize (int): Size of the batch.`
        """
        with tf.compat.v1.name_scope(name, "construct point cloud", [self, pPts, pBatchIds, pBatchSize]):
            pPts = tf.convert_to_tensor(value=pPts)
            pBatchIds = tf.convert_to_tensor(value=pBatchIds)

            self.pts_ = pPts
            self.batchIds_ = pBatchIds
            self.batchSize_ = pBatchSize

            #Sort the points based on the batch ids in incremental order.
            _, self.sortedIndicesBatch_ = tf.math.top_k(self.batchIds_, 
                tf.shape(self.batchIds_)[0])
            self.sortedIndicesBatch_ = tf.reverse(self.sortedIndicesBatch_, axis = [0])
        

    def __eq__(self, other, name=None):
        """Comparison operator.

        Args:
            other (PointCloud): Other point cloud.
        Return:
            True if other is equal to self, False otherwise.
        """
        with tf.compat.v1.name_scope(name, "compare point clouds", [self, other]):
            return self.pts_.name == other.pts_.name and \
                self.batchIds_.name == other.batchIds_.name and \
                self.batchSize_ == other.batchSize_


    def __hash__(self, name=None):
        """Method to compute the hash of the object.

        Return:
            Unique hash value.
        """
        with tf.compat.v1.name_scope(name, "hash point cloud",[self]):
            return hash((self.pts_.name, self.batchIds_.name, self.batchSize_))