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
"""helper functions for unit tests"""

import tensorflow as tf
import numpy as np


def _create_random_point_cloud_segmented(batch_size,
                                         num_points,
                                         dimension=3,
                                         sizes=None,
                                         scale=1,
                                         clean_aabb=False):
  points = np.random.uniform(0, scale, [num_points, dimension])
  if sizes is None:
    batch_ids = np.random.randint(0, batch_size, num_points)
    batch_ids[:batch_size] = np.arange(0, batch_size)
    # batch_ids = np.sort(batch_ids)
  else:
    sizes = np.array(sizes, dtype=int)
    batch_ids = np.repeat(np.arange(0, batch_size), sizes)
  if clean_aabb:
    # adds points such that the aabb is [0,0,0] [1,1,1]*scale
    # to prevent rounding errors
    points = np.concatenate(
        (points, scale * np.ones([batch_size, 3]) - 1e-9,
         1e-9 + np.zeros([batch_size, 3])))
    batch_ids = np.concatenate(
        (batch_ids, np.arange(0, batch_size), np.arange(0, batch_size)))
  return points, batch_ids


def _create_random_point_cloud_padded(max_num_points,
                                      batch_shape,
                                      dimension=3,
                                      sizes=None,
                                      scale=1,
                                      nice_aabb=False):
  batch_size = np.prod(batch_shape)
  points = np.random.uniform(
      0, scale, [max_num_points * batch_size, dimension])
  points = points.reshape(batch_shape + [max_num_points, dimension])
  if sizes is None:
    sizes = np.random.randint(1, max_num_points, batch_shape)
  return points, sizes


def _create_uniform_distributed_point_cloud_2D(num_points_sqrt,
                                               scale=1):
  ticks = np.linspace(0, scale, num=num_points_sqrt)
  points = np.array(np.meshgrid(ticks, ticks)).T
  return points
