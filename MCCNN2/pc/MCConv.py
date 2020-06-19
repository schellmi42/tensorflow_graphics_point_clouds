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
"""Class to represent point cloud convolution"""

import tensorflow as tf
from tensorflow_graphics.geometry.convolution.utils import \
    flatten_batch_to_2d


from MCCNN2.pc import AABB
from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc import Neighborhood
from MCCNN2.pc import KDEMode

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from MCCNN2Module import basis_proj


class MCConv:

  def __init__(self,
               pNumInFeatures,
               pNumOutFeatures,
               pHiddenSize,
               pNumDims,
               pConvName=None):

    with tf.compat.v1.name_scope(pConvName, "create Monte-Carlo convolution",
                                 [self, pNumOutFeatures, pNumInFeatures,
                                  pNumOutFeatures, pHiddenSize, pNumDims]):
      self.numInFeatures_ = pNumInFeatures
      self.numOutFeatures_ = pNumOutFeatures
      self.numHidden_ = pHiddenSize
      self.numDims_ = pNumDims
      if pConvName is None:
        self.convName_ = ''
      else:
        self.convName_ = pConvName

      # initialize variables
      stdDev = tf.math.sqrt(1.0 / float(self.numDims_))
      hProjVecTF = tf.compat.v1.get_variable(
          self.convName_ + '_hidden_vectors',
          shape=[self.numHidden_, self.numDims_],
          initializer=tf.initializers.TruncatedNormal(
              stddev=stdDev),
          dtype=tf.float32,
          trainable=True)
      hProjBiasTF = tf.compat.v1.get_variable(
          self.convName_ + '_hidden_biases',
          shape=[self.numHidden_, 1],
          initializer=tf.initializers.zeros(),
          dtype=tf.float32,
          trainable=True)
      self.basisTF_ = tf.concat([hProjVecTF, hProjBiasTF], axis=1)

      stdDev = tf.math.sqrt(2.0 / float(self.numHidden_ * self.numInFeatures_))
      self.weights_ = \
          tf.compat.v1.get_variable(
              self.convName_ + '_conv_weights',
              shape=[self.numHidden_ * \
                     self.numInFeatures_,
                     self.numOutFeatures_],
              initializer=tf.initializers.TruncatedNormal(
                  stddev=stdDev),
              dtype=tf.float32, trainable=True)

  def __call__(self,
               pInFeatures,
               pInPC,
               pOutPC,
               pRadius,
               pBandWidth=0.2,
               name=None):

    with tf.compat.v1.name_scope(name, "Monte-Carlo_convolution",
                                 [pInFeatures, pInPC, pOutPC, pRadius,
                                  pBandWidth]):
      pInFeatures = tf.convert_to_tensor(value=pInFeatures, dtype=tf.float32)
      if len(pInFeatures.shape) > 2:
        pInFeatures, _ = flatten_batch_to_2d(pInFeatures, pInPC.sizes_)
      pRadius = tf.convert_to_tensor(value=pRadius, dtype=tf.float32)
      pBandWidth = tf.convert_to_tensor(value=pBandWidth)

      #Create the radii tensor.
      # curRadii = np.array([pRadius for i in range(self.numDims_)])
      # radiiTensor = tf.convert_to_tensor(curRadii, np.float32)
      radiiTensor = tf.repeat([pRadius], self.numDims_)
      #Create the badnwidth tensor.
      # curBandWidth = np.concatenate([0.2 for i in range(self.numDims_)])
      # bwTensor = tf.convert_to_tensor(curBandWidth, np.float32)
      bwTensor = tf.repeat(pBandWidth, self.numDims_)

      #Compute the AABB.
      aabbIn = AABB(pInPC)

      #Compute the grid.
      grid = Grid(pInPC, aabbIn, radiiTensor)

      #Compute the neighborhood key.
      neigh = Neighborhood(grid, radiiTensor, pOutPC, 0)
      neigh.compute_pdf(bwTensor, pMode=KDEMode.constant, pPtGradients=True)

      #Compute convolution (RELU - 2, LRELU - 3, ELU - 4)
      inWeightFeat = basis_proj(neigh, pInFeatures, self.basisTF_, 3)

      #Compute the convolution.
      return tf.matmul(tf.reshape(inWeightFeat,
                                  [-1, self.numInFeatures_ * self.numHidden_]),
                       self.weights_)
