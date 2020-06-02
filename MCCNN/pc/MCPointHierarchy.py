'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCPointHierarchy.py

    \brief Python definition of the point hierarchy class.

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import sys
import numpy as np
import tensorflow as tf

from MCCNN2.pc import MCAABB
from MCCNN2.pc import MCPointCloud
from MCCNN2.pc import MCGrid
from MCCNN2.pc import MCNeighborhood
from MCCNN2.pc import MCPool 
from MCCNN2.pc import MCPoolMode

class MCPointHierarchy:
    """Class to represent a point cloud hierarchy.

    Attributes:
        aabb_ (MCAABB): Bounding box of the point cloud.
        pointClouds_ (array of MCPointCloud): List of point clouds.
        poolOps_ (arraz of MCPool): List of pooling operations used to 
            create the point hierarchy.
        
    """

    def __init__(self, pPointCloud, pCellSizes, pPoolMode = MCPoolMode.pd):
        """Constructor.

        Args:
            pPointCloud (MCPointCloud): Input point cloud.
            pCellSizes (array of numpy arrays of floats): List of cell sizes for 
                each dimension.  
            pPoolMode (MCPoolMode): Mode used to pool the points.
        """
        #Initialize the attributes.
        self.aabb_ = MCAABB(pPointCloud)
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

            curGrid = MCGrid(curPC, self.aabb_, cellSizesTensor)
            curNeighborhood = MCNeighborhood(curGrid, cellSizesTensor)
            curPoolOp = MCPool(curNeighborhood, pPoolMode)
            
            self.poolOps_.append(curPoolOp)
            self.pointClouds_.append(curPoolOp.poolPointCloud_)
            curPC = curPoolOp.poolPointCloud_
            