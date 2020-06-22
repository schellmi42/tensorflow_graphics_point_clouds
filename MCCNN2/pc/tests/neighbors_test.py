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
"""Class to point sampling operation"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from MCCNN2.pc import PointCloud
from MCCNN2.pc import AABB
from MCCNN2.pc import Grid
from MCCNN2.pc import Sample
from MCCNN2.pc import SampleMode
from MCCNN2.pc import Neighborhood
from MCCNN2.pc.tests import utils


class NeighborsTest(test_case.TestCase):

  @parameterized.parameters(
    (1000, 10, 8, [0.025, 0.025, 0.025])
  )
  def test_find_neighbors(self, num_points, num_samples,
                          batch_size, cell_sizes):
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points * batch_size,
        sizes=np.ones(batch_size, dtype=int) * num_points)
    point_cloud = PointCloud(points, batch_ids)
    samples_points, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples * batch_size,
            sizes=np.ones(batch_size, dtype=int) * num_samples)
    point_cloud_sampled = PointCloud(samples_points, batch_ids_samples)
    aabb = AABB(point_cloud)
    grid = Grid(point_cloud, aabb, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_sampled)
    sorted_points = grid.sortedPts_

    neighbors_tf = neighborhood.neighbors_

    neighbors_numpy = [[] for i in range(num_samples * batch_size)]

    for k in range(batch_size):
      for i in range(num_samples):
        for j in range(num_points):
          diffArray = (samples_points[i + k * num_samples] - \
                       sorted_points[(batch_size - k - 1) * num_points + j])\
                       / cell_sizes
          if np.linalg.norm(diffArray) < 1.0:
            neighbors_numpy[k * num_samples + i].append((batch_size - k - 1)\
                                                        * num_points + j)

    allFound = True
    for neigh in neighbors_tf:
      found = False
      for ref_neigh in neighbors_numpy[neigh[1]]:
        if ref_neigh == neigh[0]:
          found = True
        allFound = allFound and found
    self.assertTrue(allFound)

if __name__ == '__main__':
  test_case.main()
