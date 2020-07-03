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
""" modules for neighborhoods in point clouds """


import enum
import tensorflow as tf

from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc.custom_ops import find_neighbors, compute_pdf


class KDEMode(enum.Enum):
  """ Parameters for kernel density estimation (KDE) """
  constant = 0
  numPts = 1
  noPDF = 2


class Neighborhood:
  """Class to represent a neighborhood of points.

  Note:
    In the following D is the spatial dimensionality of the points,
    N is the number of (samples) points, and M is the total number of
    adjacencies.

  Attributes:
    pcSamples_: 'PointCloud', samples point cloud.
    grid_ : 'Grid', regular grid data structure.
    radii_: float 'Tensor' of shape [D], radii used to select the neighbors.
    samplesNeighRanges_: int 'Tensor' of shape [N], end of the ranges per
      sample.
    neighbors_: int 'Tensor' of shape [M,2], indices of the neighbor point and
      the sample for each neighbor.
    pdf_: float 'Tensor' of shape [M], PDF value for each neighbor.
  """

  def __init__(self,
               pGrid: Grid,
               pRadii,
               pPCSample=None,
               pMaxNeighbors=0,
               name=None):
    """Constructor.

    Args:
      pGrid: A 'Grid' instance, the regular grid data structure.
      pRadii: A float 'Tensor' of shape [D], the radii used to select the
        neighbors.
      pPCSample: A 'PointCloud' instance. Samples point cloud. If None, the
        sorted points from the grid will be used.
      pMaxNeighbors: Integer, maximum number of neighbors per sample.
    """
    with tf.compat.v1.name_scope(
        name, "constructor for neighbourhoods of point clouds",
        [self, pGrid, pRadii, pPCSample, pMaxNeighbors]):
      pRadii = tf.convert_to_tensor(value=pRadii, dtype=tf.float32)

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

  def compute_pdf(self, pBandwidth, pMode=0, name=None):
    """Method to compute the probability density function of a neighborhood.

    Args:
      pBandwidth: float 'Tensor' of shape [D], bandwidth used to compute
        the pdf.
      pMode: 'KDEMode', mode used to determine the bandwidth.
    """
    with tf.compat.v1.name_scope(
        name, "compute pdf for neighbours",
        [self, pBandwidth, pMode]):
      pBandwidth = tf.convert_to_tensor(value=pBandwidth)

      if pMode == KDEMode.noPDF:
        self.pdf_ = tf.ones_like(
            self.neighbors_[:, 0], dtype=tf.float32)
      else:
        if self.equalSamples_:
          auxNeigh = self
        else:
          auxNeigh = Neighborhood(self.grid_, self.radii_, None)
        tmpPDF = compute_pdf(
              auxNeigh, pBandwidth, pMode.value)
        self.pdf_ = tf.gather(tmpPDF, self.neighbors_[:, 0])
