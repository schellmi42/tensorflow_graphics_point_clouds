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
from tensorflow_graphics.util import test_case

from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc.tests import utils


class GridTest(test_case.TestCase):

  @parameterized.parameters(
    # (10000, 32, 30, 0.1, 2), # currently corrupted in 2D
    # (20000, 16, 1, 0.2, 2),
    (20000, 8, 1, np.sqrt(2), 2),
    (10000, 32, 30, 0.1, 3),
    (20000, 16, 1, 0.2, 3),
    (20000, 8, 1, np.sqrt(3), 3),
    (10000, 32, 30, 0.1, 4),
    (20000, 16, 1, 0.2, 4),
    (20000, 8, 1, np.sqrt(4), 4)
  )
  def test_compute_keys_with_sort(self,
                                  num_points,
                                  batch_size,
                                  scale,
                                  radius,
                                  dimension):
    radius = np.repeat(radius, dimension)
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        sizes=np.ones(batch_size, dtype=int) * num_points, clean_aabb=True)
    point_cloud = PointCloud(points, batch_ids)
    aabb = point_cloud.get_AABB()
    grid = Grid(point_cloud, aabb, radius)

    total_num_cells = grid.numCells_.numpy()
    aabb_min = aabb.aabbMin_.numpy()

    aabb_min_per_point = aabb_min[batch_ids, :]
    cell_ind = np.floor((points - aabb_min_per_point) / radius).astype(int)
    cell_ind = np.minimum(np.maximum(cell_ind, [0] * dimension),
                          total_num_cells)
    if dimension == 2:
      keys = batch_ids * total_num_cells[0] * \
          total_num_cells[1] + \
          cell_ind[:, 0] * total_num_cells[1]
    elif dimension == 3:
      keys = batch_ids * total_num_cells[0] * \
          total_num_cells[1] * total_num_cells[2] + \
          cell_ind[:, 0] * total_num_cells[1] * total_num_cells[2] + \
          cell_ind[:, 1] * total_num_cells[2] + cell_ind[:, 2]
    elif dimension == 4:
      keys = batch_ids * total_num_cells[0] * \
          total_num_cells[1] * total_num_cells[2]  * total_num_cells[3] + \
          cell_ind[:, 0] * total_num_cells[1] * total_num_cells[2] * \
          total_num_cells[3] + \
          cell_ind[:, 1] * total_num_cells[1] * total_num_cells[2] + \
          cell_ind[:, 2] * total_num_cells[1] + cell_ind[:, 3]
    # check unsorted keys
    self.assertAllEqual(grid.curKeys_, keys)

    # sort descending
    sorted_keys = np.flip(np.sort(keys))
    # check if the cell keys per point are equal
    self.assertAllEqual(grid.sortedKeys_, sorted_keys)

  @parameterized.parameters(
    # currently numpy implementation only in 3D
    # (10000, 32, 30, 0.1, 2),
    # (20000, 16, 1, 0.2, 2),
    # (20000, 8, 1, np.sqrt(2), 2),
    (10000, 32, 30, 0.1, 3),
    (20000, 16, 1, 0.2, 3),
    (20000, 8, 1, np.sqrt(3), 3),
    # (10000, 32, 30, 0.1, 4),
    # (20000, 16, 1, 0.2, 4),
    # (20000, 8, 1, np.sqrt(4), 4)
  )
  def test_grid(self, num_points, batch_size, scale, radius, dimension):
    radius = np.repeat(radius, dimension)
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        sizes=np.ones(batch_size, dtype=int) * num_points, clean_aabb=True)
    point_cloud = PointCloud(points, batch_ids)
    aabb = point_cloud.get_AABB()
    grid = Grid(point_cloud, aabb, radius)

    total_num_cells = grid.numCells_.numpy()
    aabb_min = aabb.aabbMin_.numpy()
    aabb_min_per_point = aabb_min[batch_ids, :]
    cell_ind = np.floor((points - aabb_min_per_point) / radius).astype(int)
    cell_ind = np.minimum(np.maximum(cell_ind, [0] * dimension),
                          total_num_cells)
    if dimension == 2:
      keys = batch_ids * total_num_cells[0] * \
          total_num_cells[1] + \
          cell_ind[:, 0] * total_num_cells[1]
    elif dimension == 3:
      keys = batch_ids * total_num_cells[0] * \
          total_num_cells[1] * total_num_cells[2] + \
          cell_ind[:, 0] * total_num_cells[1] * total_num_cells[2] + \
          cell_ind[:, 1] * total_num_cells[2] + cell_ind[:, 2]
    elif dimension == 4:
      keys = batch_ids * total_num_cells[0] * \
          total_num_cells[1] * total_num_cells[2]  * total_num_cells[3] + \
          cell_ind[:, 0] * total_num_cells[1] * total_num_cells[2] * \
          total_num_cells[3] + \
          cell_ind[:, 1] * total_num_cells[1] * total_num_cells[2] + \
          cell_ind[:, 2] * total_num_cells[1] + cell_ind[:, 3]

    keys = np.flip(np.sort(keys))
    # check if the cell keys per point are equal
    self.assertAllEqual(grid.sortedKeys_, keys)

    ds_numpy = np.full((batch_size, total_num_cells[0],
                        total_num_cells[1], 2), 0)

    for key_iter, key in enumerate(keys):
      curDSIndex = key // total_num_cells[2]
      yIndex = curDSIndex % total_num_cells[1]
      auxInt = curDSIndex // total_num_cells[1]
      xIndex = auxInt % total_num_cells[0]
      curbatch_ids = auxInt // total_num_cells[0]

      if key_iter == 0:
        ds_numpy[curbatch_ids, xIndex, yIndex, 0] = key_iter
      else:
        prevKey = keys[key_iter - 1]
        prevDSIndex = prevKey // total_num_cells[2]
        if prevDSIndex != curDSIndex:
            ds_numpy[curbatch_ids, xIndex, yIndex, 0] = key_iter

      nextIter = key_iter + 1
      if nextIter >= len(keys):
        ds_numpy[curbatch_ids, xIndex, yIndex, 1] = len(keys)
      else:
        nextKey = keys[key_iter + 1]
        nextDSIndex = nextKey // total_num_cells[2]
        if nextDSIndex != curDSIndex:
          ds_numpy[curbatch_ids, xIndex, yIndex, 1] = key_iter + 1

    # check if the data structure is equal
    self.assertAllEqual(grid.fastDS_, ds_numpy)

if __name__ == '__main__':
  test_case.main()
