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
from tensorflow_graphics.geometry.convolution import utils

utils.flatten_batch_to_2d

class PointCloud:
    """Class to represent a point cloud.

    Attributes:
        pts_ (float tensor nxd): List of points.
        batchIds_ (int tensor n): List of batch ids associated with the points.
        batchSize_ (int): Size of the batch.
    """

    def __init__(self, pPts, pBatchIds=None, pBatchSize=None, sizes=None, name=None):
        """Constructor.

        Args:
            pPts: A float tensor either of shape [N,D]  
                or of shape [A1,..,An,V,D], possibly padded as indicated by batch ids. Represents the point coordinates.
            pBatchIds: Either an int tensor either of shape [N] or of shape [A1,..,An,V] where padding is indicated by negative values. 
                int batch ids associated with the points of
            sizes:      An `int` tensor of shape `[A1, ..., An]` indicating the true input
                sizes in case of padding (`sizes=None` indicates no padding).Note that
                `sizes[A1, ..., An] <= V`.
            pBatchSize (int): Size of the batch.`
        """
        with tf.compat.v1.name_scope(name, "construct point cloud", [self, pPts, pBatchIds, pBatchSize, sizes]):
            pPts = tf.convert_to_tensor(value=pPts, dtype=tf.float32)
            sizes = tf.convert_to_tensor(value=sizes)
            if pBatchIds!= None:
                pBatchIds = tf.convert_to_tensor(value=pBatchIds, dtype=tf.int32)
            
            self.batchShape_ = None

            if len(pPts.shape)==1:
               raise ValueError('Point tensor is of dimension 1 but should be at least 2!')
            self.dimension_ = pPts.shape[-1]

            if sizes != None:
                self.batchShape_ = pPts.shape[:-2]
                if pBatchSize == None:
                    pBatchSize = tf.reduce_prod(self.batchShape_)
                pPts, self.unflatten = utils.flatten_batch_to_2d(pPts, sizes)
                pBatchIds = tf.repeat(tf.range(0,pBatchSize),repeats=tf.reshape(sizes,[-1]))
            elif pBatchSize == None:
                pBatchSize = tf.maximum(pBatchIds)

            self.pts_ = pPts
            self.batchIds_ = pBatchIds
            self.batchSize_ = pBatchSize

            #Sort the points based on the batch ids in incremental order.
            _, self.sortedIndicesBatch_ = tf.math.top_k(self.batchIds_, 
                tf.shape(self.batchIds_)[0])
            self.sortedIndicesBatch_ = tf.reverse(self.sortedIndicesBatch_, axis = [0])
    
    def get_points(self,id=None,name=None):
        """ Returns the points 

        Args:
            id (int): Identifier of the batch if None return all
        
        Return:
            tensor: Point of the specified batch id
        """
        with tf.compat.v1.name_scope(name, "get points", [self, id]):
            if id != None:
                return self.pts_[self.batchIds_==id]
            else:
                 return self.unflatten(self.pts_)
    
        

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