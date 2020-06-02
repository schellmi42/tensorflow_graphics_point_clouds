'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCPool.py

    \brief Python definition of a pooled point cloud.

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
import sys
import enum
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from MCCNN2Module import compute_keys
from MCCNN2Module import pooling

from MCCNN2.pc import MCPointCloud

class MCPoolMode(enum.Enum):
    pd = 0
    avg = 1

class MCPool:
    """Class to represent a pool operation on point clouds.

    Attributes:
        neighborhood_ (MCNeighborhood): Neighborhood. The samples should
            be the same as the sorted points.
        indices_ (int tensor): List of the indices of the selected points.
            Only valid for the poisson disk sampling algorithm.
        poolPointCloud_ (MCPointCloud): Pooled point cloud.
        poolMode_ (MCPoolMode): Mode used to pool points.
    """

    def __init__(self, pNeighborhood, pPoolMode = MCPoolMode.pd):
        """Constructor.

        Args:
            pNeighborhood (MCNeighborhood): Neighborhood.
            pPoolMode (MCPoolMode): Mode used to pool points.
        """

        #Save the attributes.
        self.neighborhood_ = pNeighborhood
        self.poolMode_ = pPoolMode

        #Compute the pooling.
        poolPts, poolBatchIds, poolIndices = pooling(
            self.neighborhood_, self.poolMode_.value)

        #Save the pooled point cloud.
        if pPoolMode == MCPoolMode.pd:
            self.indices_ = tf.gather(self.neighborhood_.grid_.sortedIndices_, poolIndices)
        else:
            self.indices_ = None
        self.poolPointCloud_ = MCPointCloud(poolPts, poolBatchIds, 
            self.neighborhood_.pcSamples_.batchSize_)
