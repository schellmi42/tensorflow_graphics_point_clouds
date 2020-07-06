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
""" class to represent a sampling operation """


import enum
import tensorflow as tf

from MCCNN2.pc.custom_ops import compute_keys, sampling

from MCCNN2.pc import PointCloud


class SampleMode(enum.Enum):
  pd = 0
  avg = 1


class Sample:
  """Class to represent a sample operation on point clouds.

  Attributes:
    _neighborhood (Neighborhood): Neighborhood. The samples should
      be the same as the sorted points.
    _indices (int tensor): List of the indices of the selected points.
      Only valid for the poisson disk sampling algorithm.
    _sample_point_cloud (PointCloud): Sampled point cloud.
    _sample_mode (SampleMode): Mode used to sample points, 1 for Poisson disk
    sampling, 0 for average
  """

  def __init__(self, neighborhood, sample_mode=SampleMode.pd, name=None):
    """Constructor.

    Args:
      neighborhood (Neighborhood): Neighborhood.
      sample_mode (SampleMode): Mode used to sample points.
    """
    with tf.compat.v1.name_scope(
        name, "sample point cloud", [self, neighborhood, sample_mode]):
      #Save the attributes.
      self._neighborhood = neighborhood
      self._sample_mode = sample_mode

      #Compute the sampleing.
      sampled_points, sampled_batch_ids, sampled_indices = sampling(
        self._neighborhood, self._sample_mode.value)

      #Save the sampleed point cloud.
      if sample_mode == SampleMode.pd:
        self._indices = tf.gather(
            self._neighborhood._grid._sorted_indices, sampled_indices)
      else:
        self._indices = None
      self._sample_point_cloud = PointCloud(
          points=sampled_points, batch_ids=sampled_batch_ids,
          batch_size=self._neighborhood._point_cloud_sampled._batch_size)
