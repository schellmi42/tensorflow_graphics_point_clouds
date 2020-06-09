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
from tensorflow_graphics.geometry.convolution.utils import flatten_batch_to_2d, unflatten_2d_to_batch
import utils


class PointCloud:
    """Class to represent a point cloud.
    The internal representation is a 2D tensor of shape [N,D] with segmentation ids of shape [N].
    Can be constructed by directly passing segmented inputs or by prividing a [A1,..,An,V,D] padded tensor with sizes indicating 

    Attributes:
        pts_ (float tensor [N,D]): List of points.
        sizes_ (int tensor [A1,,.An]): sizes of the point clouds, None if constructed using segmented input
        batchIds_ (int tensor n): List of batch ids associated with the points.
        batchSize_ (int): Size of the batch.
        batchShape_ (int tensor): [A1,..,An] original shape of the batch, None if constructed using segmented input
        unflatten_ (function): Function to reshape segmented [N,D] to [A1,...,An,V,D] 
        dimension_ (int): dimensionality of the point clouds
        get_segment_id_ (int tensor [A1,...,An]): tensor that returns the segment id given a relative id in [A1,...,An]

    """

    def __init__(self, pPts, pBatchIds=None, pBatchSize=None, sizes=None, name=None):
        """Constructor.

        Args:
            pPts: A float tensor either of shape [N,D]  
                or of shape [A1,..,An,V,D], possibly padded as indicated by sizes. Represents the point coordinates.
            pBatchIds: Anint tensor either of shape [N] associated with the points of pPts
            sizes:      An `int` tensor of shape `[A1, ..., An]` indicating the true input
                sizes in case of padding (`sizes=None` indicates no padding).Note that
                `sizes[A1, ..., An] <= V`.
            pBatchSize (int): Size of the batch.`
        """
        with tf.compat.v1.name_scope(name, "construct point cloud", [self, pPts, pBatchIds, pBatchSize, sizes]):
            pPts = tf.convert_to_tensor(value=pPts, dtype=tf.float32)
            if sizes != None:
                sizes = tf.convert_to_tensor(value=sizes)
            if pBatchIds!= None:
                pBatchIds = tf.convert_to_tensor(value=pBatchIds, dtype=tf.int32)
            
            self.sizes_ = sizes
            self.batchShape_ = None
            self.unflatten_ = None
            self.dimension_ = pPts.shape[-1]

            if len(pPts.shape) > 2:
                # converting padded [A1,...,An,V,D] tensor into a 2D tensor [N,D] with segmentation ids
                self.batchShape_ = pPts.shape[:-2]
                if pBatchSize == None:
                    self.batchSize_ = tf.reduce_prod(self.batchShape_)
                self.get_segment_id_ = tf.reshape(tf.range(0,self.batchSize_),self.batchShape_)
                self.pts_, self.unflatten_ = flatten_batch_to_2d(pPts, sizes)
                self.batchIds_ = tf.repeat(tf.range(0,self.batchSize_),repeats=tf.reshape(sizes,[-1]))
            elif pBatchIds != None:
                # if input is already 2D tensor with segmentation ids
                if pBatchSize == None:
                    self.batchSize_ = tf.reduce_max(pBatchIds)
                else:
                    self.batchSize_ = pBatchSize
                self.batchIds_ = pBatchIds
                self.pts_ = pPts
            else:
                raise ValueError('invalid input format.')
            
            # self.pts_ = pPts
            # self.batchIds_ = pBatchIds
            # self.batchSize_ = pBatchSize

            #Sort the points based on the batch ids in incremental order.
            _, self.sortedIndicesBatch_ = tf.math.top_k(self.batchIds_, 
                tf.shape(self.batchIds_)[0])
            self.sortedIndicesBatch_ = tf.reverse(self.sortedIndicesBatch_, axis = [0])
    
    def get_points(self,id=None,name=None):
        """ Returns the points.
            If called withoud specifying 'id' returns the points in padded format [A1,...,An,V,D]

        Args:
            id (int): Identifier of point cloud in the batch, if None return all
        
        Return:
            tensor:  if 'id' was given: 2D float tensor of shape 
                if 'id' not given: float tensor of shape [A1,...,An,V,D]
        """
        with tf.compat.v1.name_scope(name, "get points", [self, id]):
            if id != None:
                if len(id) != 1:
                    id = self.get_segment_id_[id]
                if id>self.batchSize_:
                    raise IndexError('batch index out of range')
                return self.pts_[self.batchIds_==id]
            else:
                return self.get_unflatten()(self.pts_)

    def get_sizes(self, name=None):
        """ Returns the sizes of the point clouds in the batch. 
            Note: Use this instead of accessing 'self.sizes_', 
            if the class was constructed using segmented input the sizes are computed in this method.

        Return:
            tensor of shape [A1,..,An]
        """
        with tf.compat.v1.name_scope(name, "get sizes", []):
            if self.sizes_== None:
                _,_,self.sizes_ = tf.unique_with_counts(self.batchIds_)
                self.sizes_ = tf.reshape(self.sizes,self.batchShape_)
            return self.sizes_
    
    def get_unflatten(self, name=None):
        """ Returns the function to unflatten the segmented points.
            Note: use this instead of accessing 'self.unflatten_',
            if the class was constructed using segmented input the unflatten method is computed in this method.

            Return:
                method to unflatten the segmented points.
        """
        with tf.compat.v1.name_scope(name, "get unflatten method", []):
            if self.unflatten == None:
                self.unflatten = unflatten_2d_to_batch(data=self.pts_,sizes=self.get_sizes())
            return self.unflatten_

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