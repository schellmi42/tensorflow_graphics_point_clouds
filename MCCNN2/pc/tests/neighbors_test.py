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

from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc import Sample
from MCCNN2.pc import SampleMode
from MCCNN2.pc import Neighborhood
from MCCNN2.pc.tests import utils


class NeighborsTest(test_case.TestCase):

  @parameterized.parameters(
    (100, 10, 8, 0.025, 2),
    (1000, 10, 8, 0.025, 3),
    (1000, 10, 8, 0.025, 4)
  )
  def test_find_neighbors(self,
                          num_points,
                          num_samples,
                          batch_size,
                          radius,
                          dimension):
    cell_sizes = np.repeat(radius, dimension)
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        sizes=np.ones(batch_size, dtype=int) * num_points)
    point_cloud = PointCloud(points, batch_ids)
    samples_points, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples * batch_size, dimension=dimension,
            sizes=np.ones(batch_size, dtype=int) * num_samples)
    point_cloud_sampled = PointCloud(samples_points, batch_ids_samples)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_sampled)
    sorted_points = grid._sorted_points

    neighbors_tf = neighborhood._neighbors

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

  @parameterized.parameters(
    (12, 100, 24, np.sqrt(2), 2),
    (32, 10000, 32, 0.7, 2),
    (32, 10000, 32, 0.1, 2),
    (12, 100, 24, np.sqrt(3), 3),
    (32, 10000, 32, 0.7, 3),
    (32, 10000, 32, 0.1, 3),
    (12, 100, 24, np.sqrt(4), 4),
    (32, 10000, 32, 0.7, 4),
    (32, 10000, 32, 0.1, 4),
  )
  def test_neighbors_are_from_same_batch(self,
                                         batch_size,
                                         num_points,
                                         num_samples,
                                         radius,
                                         dimension):
    points, batch_ids = utils._create_random_point_cloud_segmented(
      batch_size, num_points, dimension=dimension)
    samples, batch_ids_samples = utils._create_random_point_cloud_segmented(
      batch_size, num_samples, dimension=dimension)
    radius = np.float32(np.repeat([radius], dimension))

    point_cloud = PointCloud(points, batch_ids)
    point_cloud_samples = PointCloud(samples, batch_ids_samples)
    grid = Grid(point_cloud, radius)
    neighborhood = Neighborhood(grid, radius, point_cloud_samples)

    batch_ids_in = tf.gather(
        point_cloud._batch_ids, neighborhood._original_neigh_ids[:, 0])
    batch_ids_out = tf.gather(
        point_cloud_samples._batch_ids, neighborhood._original_neigh_ids[:, 1])
    batch_check = batch_ids_in == batch_ids_out

    self.assertTrue(np.all(batch_check))


if __name__ == '__main__':
  test_case.main()
