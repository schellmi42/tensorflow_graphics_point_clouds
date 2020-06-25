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
"""Class to test kernel density estimation for point clouds"""

import os
import sys
import numpy as np
from sklearn.neighbors import KernelDensity
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_MODULE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_MODULE_DIR, "tf_ops"))

from MCCNN2.pc import PointCloud
from MCCNN2.pc import Grid
from MCCNN2.pc import KDEMode
from MCCNN2.pc import Neighborhood
from MCCNN2.pc.tests import utils


class ComputePDFTest(test_case.TestCase):

  @parameterized.parameters(
    # (32, 100, 10, 0.2, 0.1, 2),
    # (32, 100, 10, 0.7, 0.1, 2),
    # (32, 100, 10, np.sqrt(2), 0.1, 2),
    (32, 100, 10, 0.2, 0.1, 3),
    (32, 100, 10, 0.7, 0.1, 3),
    (32, 100, 10, np.sqrt(3), 0.1, 3),
    (32, 100, 10, 0.2, 0.1, 4),
    # (32, 100, 10, 0.7, 0.1, 4),
    (32, 100, 10, np.sqrt(4), 0.1, 4)
  )
  def test_compute_pdf(self,
                       batch_size,
                       num_points,
                       num_samples_per_batch,
                       cell_size,
                       bandwidth,
                       dimension):
    cell_sizes = np.float32(np.repeat(cell_size, dimension))
    bandwidths = np.float32(np.repeat(bandwidth, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, batch_size * num_points, dimension,
        equal_sized_batches=True)
    samples = np.full((batch_size * num_samples_per_batch, dimension),
                      0.0, dtype=float)
    for i in range(batch_size):
      cur_choice = np.random.choice(num_points, num_samples_per_batch,
                                    replace=True)
      samples[num_samples_per_batch * i:num_samples_per_batch * (i + 1), :] = \
          points[cur_choice + i * num_points]
    samples_batch_ids = np.repeat(np.arange(0, batch_size),
                                  num_samples_per_batch)

    point_cloud = PointCloud(points, batch_ids, batch_size)
    aabb = point_cloud.get_AABB()
    grid = Grid(point_cloud, aabb, cell_sizes)

    point_cloud_samples = PointCloud(samples, samples_batch_ids, batch_size)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    neighborhood.compute_pdf(bandwidths, KDEMode.constant)

    sorted_points = grid.sortedPts_.numpy()
    sorted_batch_ids = grid.sortedBatchIds_.numpy()
    neighbor_ids = neighborhood.neighbors_
    pdf_tf = neighborhood.pdf_

    pdf_real = []
    accum_points = []
    prev_batch_i = -1
    for pt_i, batch_i in enumerate(sorted_batch_ids):
      if batch_i != prev_batch_i:
        if len(accum_points) > 0:
          test_points = np.array(accum_points)
          kde_skl = KernelDensity(bandwidth=bandwidth)
          kde_skl.fit(test_points)
          log_pdf = kde_skl.score_samples(test_points)
          pdf = np.exp(log_pdf)
          if len(pdf_real) > 0:
            pdf_real = np.concatenate((pdf_real, pdf), axis=0)
          else:
            pdf_real = pdf
        accum_points = [sorted_points[pt_i] / cell_size]
        prev_batch_i = batch_i
      else:
        accum_points.append(sorted_points[pt_i] / cell_size)

    test_points = np.array(accum_points)
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(test_points)
    log_pdf = kde_skl.score_samples(test_points)
    pdf = np.exp(log_pdf)
    if len(pdf_real) > 0:
      pdf_real = np.concatenate((pdf_real, pdf), axis=0)
    else:
      pdf_real = pdf

    pdf_tf = np.asarray(pdf_tf / float(len(accum_points)))
    pdf_skl = np.asarray(pdf_real)[neighbor_ids[:, 0]]
    self.assertAllClose(pdf_tf, pdf_skl)

  @parameterized.parameters(
    # (1, 200, 1, 4, 2),
    (1, 200, 1, 4, 3),
    (1, 200, 1, 4, 4)
  )
  def test_compute_pdf_jacobian(self,
                                batch_size,
                                num_points,
                                num_samples,
                                radius,
                                dimension):
    cell_sizes = np.float32(np.repeat(radius, dimension))
    bandwidths = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, batch_size * num_points, dimension,
        equal_sized_batches=True)
    samples = np.full((batch_size * num_samples, dimension), 0.0, dtype=float)
    for i in range(batch_size):
      cur_choice = np.random.choice(num_points, num_samples, replace=True)
      samples[num_samples * i:num_samples * (i + 1), :] = \
          points[cur_choice + i * num_points]
    samples_batch_ids = np.repeat(np.arange(0, batch_size), num_samples)
    def compute_pdf(points_in):
      point_cloud = PointCloud(points_in, batch_ids, batch_size)
      aabb = point_cloud.get_AABB()
      grid = Grid(point_cloud, aabb, cell_sizes)

      point_cloud_samples = PointCloud(samples, samples_batch_ids, batch_size)
      neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
      neighborhood.compute_pdf(bandwidths, KDEMode.constant)
      norm_factors = tf.math.unsorted_segment_sum(
          tf.ones_like(neighborhood.pdf_),
          neighborhood.neighbors_[:, 1],
          batch_size * num_samples)
      norm_pdf = neighborhood.pdf_ / tf.gather(norm_factors,
                                               neighborhood.neighbors_[:, 1])
      return norm_pdf

    self.assert_jacobian_is_correct_fn(
        compute_pdf, [np.float32(points)], atol=1e-4)


if __name__ == '__main__':
  test_case.main()
