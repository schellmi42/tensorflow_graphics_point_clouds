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
from MCCNN2.pc.utils import _flatten_features


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
  """ Class to represent a Monte-Carlo convolution layer

    Attributes:
      numInFeatures_: Integer, the number of features per input point
      numoutFeatures_: Integer, the number of features to compute
      numHidden_: Integer, the number of neurons in the hidden layer of the
        kernel MLP
      numDims_: Integer, dimensionality of the point cloud
      convName_: String, name for the operation
  """

  def __init__(self,
               pNumInFeatures,
               pNumOutFeatures,
               pHiddenSize,
               pNumDims,
               pConvName=None):
    """ Constructior, initializes weights

    Args:
    pNumInFeatures: Integer D_in, the number of features per input point
    pNumOutFeatures: Integer D_out, the number of features to compute
    pHiddenSize: Integer, the number of neurons in the hidden layer of the
        kernel MLP
    pNumDims: Integer, dimensionality of the point cloud
    pConvName: String, name for the operation
    """

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
               pInPC: PointCloud,
               pOutPC: PointCloud,
               pRadius,
               pBandWidth=0.2,
               return_sorted=False,
               name=None):
    """ Computes the Monte-Carlo Convolution

    Note:
      In the following, A1 to An are optional batch dimensions.
      D_in is the number of input features.
      D_out is the number of output features.

    Args:
      pInFeatures: A float Tensor of shape [N_in,D_in] or [A1,...,An,V,D_in],
        the size must be the same as the points in the input point cloud.
      pInPC: A PointCloud instance, represents the input point cloud.
      pOutPC: A PointCloud instance, represents the output point cloud.
      pRadius: A float, the convolution radius.
      pBandWidth: The used bandwidth used in the kernel densitiy estimation on
        the input point cloud.
      return_sorted: A boolean, if 'True' the output tensor is sorted
        according to the batch_ids.

      Returns:
        Tensor with shape [N_out,D_out]
    """

    with tf.compat.v1.name_scope(name, "Monte-Carlo_convolution",
                                 [pInFeatures, pInPC, pOutPC, pRadius,
                                  pBandWidth, return_sorted]):
      pInFeatures = tf.convert_to_tensor(value=pInFeatures, dtype=tf.float32)
      pInFeatures = _flatten_features(pInFeatures, pInPC)
      pRadius = tf.convert_to_tensor(value=pRadius, dtype=tf.float32)
      pBandWidth = tf.convert_to_tensor(value=pBandWidth)

      #Create the radii tensor.
      radiiTensor = tf.repeat([pRadius], self.numDims_)
      #Create the badnwidth tensor.
      bwTensor = tf.repeat(pBandWidth, self.numDims_)

      #Compute the AABB.
      aabbIn = AABB(pInPC)

      #Compute the grid.
      grid = Grid(pInPC, aabbIn, radiiTensor)

      #Compute the neighborhood key.
      neigh = Neighborhood(grid, radiiTensor, pOutPC)
      neigh.compute_pdf(bwTensor, pMode=KDEMode.constant, pPtGradients=True)

      #Compute convolution (RELU - 2, LRELU - 3, ELU - 4)
      inWeightFeat = basis_proj(neigh, pInFeatures, self.basisTF_, 3)

      #Compute the convolution.
      convolution_result = tf.matmul(tf.reshape(
          inWeightFeat, [-1, self.numInFeatures_ * self.numHidden_]),
          self.weights_)
      if return_sorted:
        convolution_result = tf.gather(convolution_result,
                                       pOutPC.sortedIndicesBatch_)
      return convolution_result


class MonteCarloConvolution(MCConv):
  """ Class to represent a Monte-Carlo convolution layer on one point cloud

    Attributes:
      numInFeatures_: Integer, the number of features per input point
      numoutFeatures_: Integer, the number of features to compute
      numHidden_: Integer, the number of neurons in the hidden layer of the
        kernel MLP
      numDims_: Integer, dimensionality of the point cloud
      convName_: String, name for the operation
  """

  def __init__(self,
               pNumInFeatures,
               pNumOutFeatures,
               pHiddenSize,
               pNumDims,
               pConvName=None):
    """ Constructior, initializes weights

    Args:
    pNumInFeatures: Integer D_in, the number of features per input point
    pNumOutFeatures: Integer D_out, the number of features to compute
    pHiddenSize: Integer, the number of neurons in the hidden layer of the
        kernel MLP
    pNumDims: Integer, dimensionality of the point cloud
    pConvName: String, name for the operation
    """
    super(self).__init__(pNumInFeatures, pNumOutFeatures, pHiddenSize,
                         pNumDims, pConvName)

  def __call__(self,
               pInFeatures,
               pPC: PointCloud,
               pRadius,
               pBandWidth=0.2,
               return_sorted=False,
               name=None):
    """ Computes the Monte-Carlo Convolution

    Note:
      In the following, A1 to An are optional batch dimensions.
      D_in is the number of input features.

    Args:
      pInFeatures: A float Tensor of shape [N,D_in] or [A1,...,An,V,D],
        the size must be the same as the points in the input point cloud.
      pPC: A PointCloud instance
      pRadius: A float, the convolution radius.
      pBandWidth: The used bandwidth used in the kernel densitiy estimation on
        the input point cloud.
      return_sorted: A boolean, if 'True' the output tensor is sorted
        according to the batch_ids.

      Returns:
        Tensor with shape [N,D_out]
    """
    return super(self).__call__(self, pInFeatures, pPC, pPC, pRadius,
                                pBandWidth, return_sorted, name)
