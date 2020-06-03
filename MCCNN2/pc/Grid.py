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
"""Class to represent """

import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from MCCNN2Module import compute_keys
from MCCNN2Module import build_grid_ds

class Grid:
    """Class to represent a point cloud distributed in a regular grid.

    Attributes:
        batchSize_ (int): Size of the batch.
        cellSizes_ (float tensor d): Cell size.
        pointCloud_ (PointCloud): Point cloud.
        aabb_ (AABB): AABB.
        numCells_ (int tensor d): Number of cells of the grids.
        curKeys_ (int tensor n): Keys of each point.
        sortedKeys_ (int tensor n): Keys of each point sorted.
        sortedIndices_ (int tensor n): Original indices to the original
            points.
        fastDS_ (int tensor BxCXxCY): Fast access data structure.
    """

    def __init__(self, pPointCloud, pAABB, pCellSizes, name=None):
        """Constructor.

        Args:
            pPointCloud (PointCloud): Point cloud to distribute in the grid.
            pAABB (AABB): Bounding boxes.
            pCellSizes (tensor float n): Size of the grid cells in each dimension.
        """
        with tf.compat.v1.name_scope(name, "constructor for point cloud regular grid", [self, pPointCloud, pAABB, pCellSizes]):
            pCellSizes = tf.convert_to_tensor(value=pCellSizes)
            
            #Save the attributes.
            self.batchSize_ = pAABB.batchSize_
            self.cellSizes_ = pCellSizes
            self.pointCloud_ = pPointCloud
            self.aabb_ = pAABB

            #Compute the number of cells in the grid.
            aabbSizes = self.aabb_.aabbMax_ - self.aabb_.aabbMin_
            batchNumCells = tf.cast(tf.math.ceil(aabbSizes/self.cellSizes_), tf.int32)
            self.numCells_ = tf.maximum(tf.reduce_max(batchNumCells, axis=0), 1)

            #Compute the key for each point.
            self.curKeys_ = compute_keys(self.pointCloud_, self.aabb_, self.numCells_, 
                self.cellSizes_)

            #Sort the keys.
            self.sortedKeys_, self.sortedIndices_ = tf.math.top_k(self.curKeys_, 
                tf.shape(self.curKeys_)[0])

            #Compute the invert indexs.
            _, ptIndexs = tf.math.top_k(self.sortedIndices_, 
                tf.shape(self.sortedIndices_)[0])
            self.invertedIndices_ = tf.reverse(ptIndexs, [0])

            #Get the sorted points and batch ids.
            self.sortedPts_ = tf.gather(self.pointCloud_.pts_, self.sortedIndices_)
            self.sortedBatchIds_ = tf.gather(self.pointCloud_.batchIds_, self.sortedIndices_)

            #Build the fast access data structure.
            self.fastDS_ = build_grid_ds(self.sortedKeys_, self.numCells_, self.batchSize_)

