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

from MCCNN2.pc import AABB
from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc import Neighborhood
from MCCNN2.pc import Pool 
from MCCNN2.pc import PoolMode

class PointHierarchy:
    """Class to represent a point cloud hierarchy.

    Attributes:
        aabb_ (AABB): Bounding box of the point cloud.
        pointClouds_ (array of PointCloud): List of point clouds.
        poolOps_ (arraz of Pool): List of pooling operations used to 
            create the point hierarchy.
        
    """

    def __init__(self, pPointCloud, pCellSizes, pPoolMode = PoolMode.pd, name=None):
        """Constructor.

        Args:
            pPointCloud (PointCloud): Input point cloud.
            pCellSizes (array of numpy arrays of floats): List of cell sizes for 
                each dimension.  
            pPoolMode (PoolMode): Mode used to pool the points.
        """
        with tf.compat.v1.name_scope(name,"hierarchical point cloud constructor", [self, pPointCloud, pCellSizes, pPoolMode]):

            #Initialize the attributes.
            self.aabb_ = AABB(pPointCloud)
            self.pointClouds_ = [pPointCloud]
            self.poolOps_ = []
            self.cellSizes_ = []
            
            #Compute the number of dimensions of the input point cloud.
            numDims = pPointCloud.pts_.shape[1]

            #Create the different pooling operations.
            curPC = pPointCloud
            for poolIter, curCellSizes in enumerate(pCellSizes):

                #Check if the cell size is defined for all the dimensions.
                #If not, the last cell size value is tiled until all the dimensions have a value. 
                curNumDims = curCellSizes.shape[0]
                if curNumDims < numDims:
                    curCellSizes = np.concatenate((curCellSizes, 
                        np.tile(curCellSizes[-1], numDims-curNumDims)))
                self.cellSizes_.append(curCellSizes)
                
                #Create the pooling operation.
                cellSizesTensor = tf.convert_to_tensor(curCellSizes, np.float32)

                curGrid = Grid(curPC, self.aabb_, cellSizesTensor)
                curNeighborhood = Neighborhood(curGrid, cellSizesTensor)
                curPoolOp = Pool(curNeighborhood, pPoolMode)
                
                self.poolOps_.append(curPoolOp)
                self.pointClouds_.append(curPoolOp.poolPointCloud_)
                curPC = curPoolOp.poolPointCloud_
            