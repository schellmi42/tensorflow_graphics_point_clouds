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

import os
import sys
import enum
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from MCCNN2Module import compute_keys
from MCCNN2Module import sampling

from MCCNN2.pc import PointCloud


class SampleMode(enum.Enum):
  pd = 0
  avg = 1


class Sample:
  """Class to represent a sample operation on point clouds.

  Attributes:
    neighborhood_ (Neighborhood): Neighborhood. The samples should
      be the same as the sorted points.
    indices_ (int tensor): List of the indices of the selected points.
      Only valid for the poisson disk sampling algorithm.
    samplePointCloud_ (PointCloud): Sampleed point cloud.
    sampleMode_ (SampleMode): Mode used to sample points, 1 for Poisson disk
    sampling, 0 for average
  """

  def __init__(self, pNeighborhood, pSampleMode=SampleMode.pd, name=None):
    """Constructor.

    Args:
      pNeighborhood (Neighborhood): Neighborhood.
      pSampleMode (SampleMode): Mode used to sample points.
    """
    with tf.compat.v1.name_scope(
        name, "sample point cloud", [self, pNeighborhood, pSampleMode]):
      #Save the attributes.
      self.neighborhood_ = pNeighborhood
      self.sampleMode_ = pSampleMode

      #Compute the sampleing.
      samplePts, sampleBatchIds, sampleIndices = sampling(
        self.neighborhood_, self.sampleMode_.value)

      #Save the sampleed point cloud.
      if pSampleMode == SampleMode.pd:
        self.indices_ = tf.gather(
            self.neighborhood_.grid_.sortedIndices_, sampleIndices)
      else:
        self.indices_ = None
      self.samplePointCloud_ = PointCloud(
          pPts=samplePts, pBatchIds=sampleBatchIds,
          pBatchSize=self.neighborhood_.pcSamples_.batchSize_)
