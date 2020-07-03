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
# See the License for the specific
"""Class to test regular grid data structure"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
  from tfg_custom_ops.compute_keys.python.ops.compute_keys_ops import compute_keys
except ImportError:
  import compute_keys


def _create_random_point_cloud_segmented(batch_size,
                                         num_points,
                                         dimension=3,
                                         sizes=None,
                                         scale=1,
                                         clean_aabb=False,
                                         equal_sized_batches=False):
  points = np.random.uniform(0, scale, [num_points, dimension])
  if sizes is None:
    if not equal_sized_batches:
      batch_ids = np.random.randint(0, batch_size, num_points)
      batch_ids[:batch_size] = np.arange(0, batch_size)
    else:
      batch_ids = np.repeat(np.arange(0, batch_size), num_points // batch_size)
    # batch_ids = np.sort(batch_ids)
  else:
    sizes = np.array(sizes, dtype=int)
    batch_ids = np.repeat(np.arange(0, batch_size), sizes)
  if clean_aabb:
    # adds points such that the aabb is [0,0,0] [1,1,1]*scale
    # to prevent rounding errors
    points = np.concatenate(
        (points, scale * np.ones([batch_size, dimension]) - 1e-9,
         1e-9 + np.zeros([batch_size, dimension])))
    batch_ids = np.concatenate(
        (batch_ids, np.arange(0, batch_size), np.arange(0, batch_size)))
  return points, batch_ids


class GridTest(test.TestCase):

  @test_util.run_gpu_only
  def test_compute_keys_with_sort(self):
    num_points = 10000
    batch_size = 32
    radius = 0.1
    dimension = 3
    radius = np.repeat(radius, dimension)
    points, batch_ids = _create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        sizes=np.ones(batch_size, dtype=int) * num_points, clean_aabb=True)
    aabb_min = np.amin(points, axis=0)
    aabb_sizes = np.amax(points, axis=0) - aabb_min
    total_num_cells = np.max(np.ceil(aabb_sizes / radius))
    custom_keys = compute_keys(
        points, batch_ids, aabb_min / radius, total_num_cells, 1 / radius)

    aabb_min_per_point = aabb_min[batch_ids, :]
    cell_ind = np.floor((points - aabb_min_per_point) / radius).astype(int)
    cell_ind = np.minimum(np.maximum(cell_ind, [0] * dimension),
                          total_num_cells)
    if dimension == 2:
      ref_keys = batch_ids * total_num_cells[0] * \
          total_num_cells[1] + \
          cell_ind[:, 0] * total_num_cells[1]
    elif dimension == 3:
      ref_keys = batch_ids * total_num_cells[0] * \
          total_num_cells[1] * total_num_cells[2] + \
          cell_ind[:, 0] * total_num_cells[1] * total_num_cells[2] + \
          cell_ind[:, 1] * total_num_cells[2] + cell_ind[:, 2]
    elif dimension == 4:
      ref_keys = batch_ids * total_num_cells[0] * \
          total_num_cells[1] * total_num_cells[2]  * total_num_cells[3] + \
          cell_ind[:, 0] * total_num_cells[1] * total_num_cells[2] * \
          total_num_cells[3] + \
          cell_ind[:, 1] * total_num_cells[1] * total_num_cells[2] + \
          cell_ind[:, 2] * total_num_cells[1] + cell_ind[:, 3]
    # check unsorted keys
    self.assertAllEqual(custom_keys, ref_keys)


if __name__ == '__main__':
  test.main()
