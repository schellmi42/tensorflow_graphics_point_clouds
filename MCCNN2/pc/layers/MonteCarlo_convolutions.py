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
"""Class to represent monte-carlo point cloud convolution"""

from MCCNN2.pc import MCConv
from MCCNN2.pn import PointCloud


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
    return super(self).__call__(self, pInFeatures, pInPC, pOutPC, pRadius,
                                pBandWidth, return_sorted, name)
