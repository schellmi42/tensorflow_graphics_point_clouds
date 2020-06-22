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
"""Class to represent hierarchical point clouds"""

import os
import sys
import numpy as np
import tensorflow as tf

from MCCNN2.pc.utils import check_valid_point_hierarchy_input

from MCCNN2.pc import AABB
from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc import Neighborhood
from MCCNN2.pc import Sample
from MCCNN2.pc import SampleMode


class PointHierarchy:
  """Class to represent a point cloud hierarchy.

  Attributes:
    aabb_ (AABB): Bounding box of the point cloud.
    pointClouds_ (array of PointCloud): List of point clouds.
    sampleOps_ (arraz of Sample): List of sampling operations used to
      create the point hierarchy.
  """

  def __init__(self,
               pPointCloud,
               pCellSizes,
               pSampleMode=SampleMode.pd,
               name=None):
    """Constructor.

    Args:
      pPointCloud (PointCloud): Input point cloud.
      pCellSizes (array of numpy arrays of floats): List of cell sizes for
        each dimension.
      pSampleMode (SampleMode): Mode used to sample the points.
    """
    with tf.compat.v1.name_scope(
        name, "hierarchical point cloud constructor",
        [self, pPointCloud, pCellSizes, pSampleMode]):

      # check_valid_point_hierarchy_input(pPointCloud,pCellSizes,pSampleMode)

      #Initialize the attributes.
      self.aabb_ = AABB(pPointCloud)
      self.pointClouds_ = [pPointCloud]
      self.sampleOps_ = []
      self.cellSizes_ = []

      self.dimensions_ = pPointCloud.dimension_
      self.batchShape_ = pPointCloud.batchShape_

      #Create the different sampling operations.
      curPC = pPointCloud
      for sampleIter, curCellSizes in enumerate(pCellSizes):
        curCellSizes  = tf.convert_to_tensor(curCellSizes, dtype=tf.float32)

        # Check if the cell size is defined for all the dimensions.
        # If not, the last cell size value is tiled until all the dimensions
        # have a value.
        curNumDims = curCellSizes.shape[0]
        if curNumDims < self.dimensions_:
          curCellSizes = np.concatenate(
              (curCellSizes, np.tile(curCellSizes[-1],
                                     self.dimensions_ - curNumDims)))
        elif curNumDims > self.dimensions_:
          raise ValueError(
              f'Too many dimensions in cell sizes {curNumDims} \
                instead of max. {self.dimensions_}')
        self.cellSizes_.append(curCellSizes)

        #Create the sampling operation.
        cellSizesTensor = tf.convert_to_tensor(curCellSizes, np.float32)

        curGrid = Grid(curPC, self.aabb_, cellSizesTensor)
        curNeighborhood = Neighborhood(curGrid, cellSizesTensor)
        curSampleOp = Sample(curNeighborhood, pSampleMode)

        self.sampleOps_.append(curSampleOp)
        curSampleOp.sampledPointCloud_.set_batch_shape(self.batchShape_)
        self.pointClouds_.append(curSampleOp.sampledPointCloud_)
        curPC = curSampleOp.sampledPointCloud_

  def get_points(self, id=None, max_num_points=None, name=None):
    """ Returns the points.

    Note:
      In the following, A1 to An are optional batch dimensions.

      If called withoud specifying 'id' returns the points in padded format
      [A1,...,An,V,D]

    Args:
      id Identifier of point cloud in the batch, if None return all points

    Return:
      list of tensors:  if 'id' was given: 2D float tensors,
        if 'id' not given: float tensors of shape [A1,...,An,V,D].
    """
    with tf.compat.v1.name_scope(
        name, "get points of specific batch id", [self, id]):
      points = []
      for point_cloud in self.pointClouds_:
        points.append(point_cloud.get_points(id))
      return points

  def get_sizes(self, name=None):
    """ Returns the sizes of the point clouds in the point hierarchy.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Return:
      list of tensors of shape [A1,..,An]
    """

    with tf.compat.v1.name_scope(name, "get point hierarchy sizes", [self]):
      sizes = []
      for point_cloud in self.pointClouds_:
        sizes.append(point_cloud.get_sizes())
      return sizes

  def set_batch_shape(self, batchShape, name=None):
    """ Function to change the batch shape

      Use this to set a batch shape instead of using 'self.batchShape_'
      to also change dependent variables.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      batchShape: float tensor of shape [A1,...,An]

    Raises:
      ValueError: if shape does not sum up to batch size.
    """
    with tf.compat.v1.name_scope(
        name, "set batch shape of point hierarchy", [self, batchShape]):
      for point_cloud in self.pointClouds_:
        point_cloud.set_batch_shape(batchShape)

  def __getitem__(self, index):
    return self.pointClouds_[index]

  def __len__(self):
    return len(self.pointClouds_)
