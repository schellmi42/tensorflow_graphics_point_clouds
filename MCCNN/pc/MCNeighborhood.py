'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCNeighborhood.py

    \brief Python definition of a neighborhood of points.

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

from MCCNN2.pc import MCPointCloud

from MCCNN2Module import find_neighbors
from MCCNN2Module import compute_pdf
from MCCNN2Module import compute_pdf_with_pt_grads
from MCCNN2Module import compute_smooth_weights
from MCCNN2Module import compute_smooth_weights_with_pt_grads

class MCKDEMode(enum.Enum):
    constant = 0
    numPts = 1
    noPDF = 2

class MCNeighborhood:
    """Class to represent a neighborhood of points.

    Attributes:
        pcSamples_ (MCPointCloud): Samples point cloud.
        grid_  (MCGrid): Regular grid data structure.
        radii_ (float tensor d): Radii used to select the neighbors.
        samplesNeighRanges_ (int tensor n): End of the ranges for each sample.
        neighbors_ (int tensor mx2): Indices of the neighbor point and the sample
            for each neighbor.
        pdf_ (float tensor m): PDF value for each neighbor.
    """

    def __init__(self, pGrid, pRadii, pPCSample = None, pMaxNeighbors = 0):
        """Constructor.

        Args:
            pGrid  (MCGrid): Regular grid data structure.
            pRadii (float tensor d): Radii used to select the neighbors.
            pPCSample (MCPointCloud): Samples point cloud. If None, the sorted
                points from the grid will be used.
            pMaxNeighbors (int): Maximum number of neighbors per sample.
        """

        #Save the attributes.
        if pPCSample is None:
            self.equalSamples_ = True
            self.pcSamples_ = MCPointCloud(pGrid.sortedPts_, \
                pGrid.sortedBatchIds_, pGrid.batchSize_)
        else:
            self.equalSamples_ = False
            self.pcSamples_ = pPCSample
        self.grid_ = pGrid
        self.radii_ = pRadii
        self.pMaxNeighbors_ = pMaxNeighbors

        #Find the neighbors.
        self.samplesNeighRanges_, self.neighbors_ = find_neighbors(
            self.grid_, self.pcSamples_, self.radii_, pMaxNeighbors)

        #Original neighIds.
        auxOriginalNeighsIds = tf.gather(self.grid_.sortedIndices_, self.neighbors_[:,0])
        self.originalNeighIds_ = tf.concat([
            tf.reshape(auxOriginalNeighsIds, [-1,1]), 
            tf.reshape(self.neighbors_[:,1], [-1,1])], axis=-1)

        #Initialize the pdf and smooth weights.
        self.pdf_ = None
        self.smoothW_ = None
        

    def compute_pdf(self, pBandwidth, pMode = 0, pPtGradients = False):
        """Method to compute the pdf of a neighborhood.

        Args:
            pBandwidth (float tensor d): Bandwdith used to compute the pdf.
            pMode (MCKDEMode): Mode used to determine the bandwidth.
            pPtGradients (bool): Boolean that determines if the operation
                will compute gradients for the input points or not.
        """
        if pMode == MCKDEMode.noPDF:
            self.pdf_ = tf.ones_like(self.neighbors_[:, 0], dtype=tf.float32)
        else:
            if self.equalSamples_:
                auxNeigh = self
            else:
                auxNeigh = MCNeighborhood(self.grid_, self.radii_, None)
            if pPtGradients:
                tmpPDF = compute_pdf_with_pt_grads(auxNeigh, pBandwidth, pMode.value)
            else:
                tmpPDF = compute_pdf(auxNeigh, pBandwidth, pMode.value)
            self.pdf_ = tf.gather(tmpPDF, self.neighbors_[:, 0])


    # def compute_smooth_weights(self, pPtGradients = False):
    #     """Method to compute the smooth weights of a neighborhood.

    #     Args:
    #         pPtGradients (bool): Boolean that determines if the operation
    #             will compute gradients for the input points or not.
    #     """
    #     if pPtGradients:
    #         self.smoothW_ = compute_smooth_weights_with_pt_grads(self, self.radii_)
    #     else:
    #         self.smoothW_ = compute_smooth_weights(self, self.radii_)


    def apply_neighbor_mask(self, pMask):
        """Method to apply a mask to the neighbors.

        Args:
            pMask (bool tensor n): Tensor with a bool element for each neighbor.
                Those which True will remain in the nieghborhood.

        """

        #Compute the new neighbor list.
        indices = tf.reshape(tf.where(pMask), [-1])
        self.neighbors_ = tf.gather(self.neighbors_, indices)
        self.originalNeighIds_ = tf.gather(
            self.originalNeighIds_, indices)
        newNumNeighs = tf.math.unsorted_segment_sum(
            tf.ones_like(self.neighbors_), 
            self.neighbors_[:,1], 
            tf.shape(self.samplesNeighRanges_)[0])
        self.samplesNeighRanges_ = tf.math.cumsum(newNumNeighs)

        #Update the smooth values.
        if not(self.smoothW_ is None):
            self.smoothW_ = tf.gather(self.smoothW_, indices)

        #Update the pdf values.
        if not(self.pdf_ is None):
            self.pdf_ = tf.gather(self.pdf_, indices)
