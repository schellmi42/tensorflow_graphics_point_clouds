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
""" modules for neighborhoods in point coulds """

import os
import sys
import enum
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from MCCNN2.pc import PointCloud
from MCCNN2Module import find_neighbors
from MCCNN2Module import compute_pdf
from MCCNN2Module import compute_pdf_with_pt_grads


class KDEMode(enum.Enum):
  """ Parameters for kernel density estimation (KDE) """
  constant = 0
  numPts = 1
  noPDF = 2


class Neighborhood:
  """Class to represent a neighborhood of points.

  Attributes:
    pcSamples_ (PointCloud): Samples point cloud.
    grid_  (Grid): Regular grid data structure.
    radii_ (float tensor d): Radii used to select the neighbors.
    samplesNeighRanges_ (int tensor n): End of the ranges for each sample.
    neighbors_ (int tensor mx2): Indices of the neighbor point and the
      sample for each neighbor.
    pdf_ (float tensor m): PDF value for each neighbor.
  """

  def __init__(self,
               pGrid,
               pRadii,
               pPCSample=None,
               pMaxNeighbors=0,
               name=None):
    """Constructor.

    Args:
      pGrid  (Grid): Regular grid data structure.
      pRadii (float tensor d): Radii used to select the neighbors.
      pPCSample (PointCloud): Samples point cloud. If None, the sorted
        points from the grid will be used.
      pMaxNeighbors (int): Maximum number of neighbors per sample.
    """
    with tf.compat.v1.name_scope(
        name, "constructor for neighbourhoods of point clouds",
        [self, pGrid, pRadii, pPCSample, pMaxNeighbors]):
      pRadii = tf.convert_to_tensor(value=pRadii)

      #Save the attributes.
      if pPCSample is None:
        self.equalSamples_ = True
        self.pcSamples_ = PointCloud(
            pGrid.sortedPts_, pGrid.sortedBatchIds_,
            pGrid.batchSize_)
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
      auxOriginalNeighsIds = tf.gather(
          self.grid_.sortedIndices_, self.neighbors_[:, 0])
      self.originalNeighIds_ = tf.concat([
        tf.reshape(auxOriginalNeighsIds, [-1, 1]),
        tf.reshape(self.neighbors_[:, 1], [-1, 1])], axis=-1)

      #Initialize the pdf
      self.pdf_ = None

  def compute_pdf(self, pBandwidth, pMode=0, pPtGradients=True, name=None):
    """Method to compute the probability density function of a neighborhood.

    Args:
      pBandwidth (float tensor d): Bandwidth used to compute the pdf.
      pMode (KDEMode): Mode used to determine the bandwidth.
      pPtGradients (bool): Boolean that determines if the operation
        will compute gradients for the input points or not.
    """
    with tf.compat.v1.name_scope(
        name, "compute pdf for neighbours",
        [self, pBandwidth, pMode, pPtGradients]):
      pBandwidth = tf.convert_to_tensor(value=pBandwidth)

      if pMode == KDEMode.noPDF:
        self.pdf_ = tf.ones_like(
            self.neighbors_[:, 0], dtype=tf.float32)
      else:
        if self.equalSamples_:
          auxNeigh = self
        else:
          auxNeigh = Neighborhood(self.grid_, self.radii_, None)
        if pPtGradients:
          tmpPDF = compute_pdf_with_pt_grads(
              auxNeigh, pBandwidth, pMode.value)
        else:
          tmpPDF = compute_pdf(auxNeigh, pBandwidth, pMode.value)
        self.pdf_ = tf.gather(tmpPDF, self.neighbors_[:, 0])
