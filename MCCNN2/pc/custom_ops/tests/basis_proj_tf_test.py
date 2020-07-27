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

from MCCNN2.pc import PointCloud, Grid, Neighborhood, AABB
from MCCNN2.pc.tests import utils
from MCCNN2.pc.layers import MCConv
from MCCNN2.pc.custom_ops.custom_ops_tf import basis_proj_tf


class BasisProjTFTest(test_case.TestCase):

  @parameterized.parameters(
    # neighbor ids are currently corrupted on dimension 2: todo fix
    # (2000, 200, [3, 3], 16, 0.7, 4, 2),
    # (4000, 400, [3, 3], 8, np.sqrt(2), 8, 2),
    (2000, 200, [1, 3], 16, 0.7, 8, 3),
    (4000, 400, [3, 3], 8, 0.7, 8, 3),
    (4000, 100, [3, 1], 1, np.sqrt(3), 16, 3),
    (2000, 200, [3, 3], 16, 0.7, 8, 4),
    (4000, 400, [1, 3], 8, np.sqrt(4), 32, 4)
  )
  def test_basis_proj(self,
                      num_points,
                      num_samples,
                      num_features,
                      batch_size,
                      radius,
                      hidden_size,
                      dimension):
    cell_sizes = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points, dimension=dimension)
    features = np.random.rand(num_points, num_features[0])
    point_cloud = PointCloud(points, batch_ids)

    point_samples, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples, dimension=dimension)

    point_cloud_samples = PointCloud(point_samples, batch_ids_samples)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    bandwidth = np.float32(np.repeat(0.2, dimension))
    pdf = neighborhood.get_pdf(bandwidth=bandwidth)
    nb_ids = neighborhood._original_neigh_ids
    # tf
    conv_layer = MCConv(
        num_features[0], num_features[1], dimension, hidden_size)

    neigh_point_coords = points[nb_ids[:, 0]]
    center_point_coords = point_samples[nb_ids[:, 1]]
    kernel_input = (neigh_point_coords - center_point_coords) / radius

    weighted_latent_per_sample_tf = basis_proj_tf(kernel_input,
                                                  neighborhood,
                                                  pdf,
                                                  features,
                                                  conv_layer._basis_tf,
                                                  2)

    # numpy
    pdf = neighborhood.get_pdf().numpy()
    neighbor_ids = neighborhood._original_neigh_ids.numpy()
    nb_ranges = neighborhood._samples_neigh_ranges.numpy()
    # extract variables
    hidden_weights = conv_layer._basis_tf[:, :-1].numpy()
    hidden_biases = conv_layer._basis_tf[:, -1].numpy()
    hidden_biases = np.expand_dims(hidden_biases, 1)

    features_on_neighbors = features[neighbor_ids[:, 0]]
    # compute first layer of kernel MLP
    point_diff = (points[neighbor_ids[:, 0]] -\
                  point_samples[neighbor_ids[:, 1]])\
        / np.expand_dims(cell_sizes, 0)
    latent_per_nb = np.dot(hidden_weights, point_diff.T) + hidden_biases
    latent_relu_per_nb = np.maximum(latent_per_nb, 0)
    # Monte-Carlo integration after first layer
    # weighting with pdf
    weighted_features_per_nb = np.expand_dims(features_on_neighbors, 2) * \
        np.expand_dims(latent_relu_per_nb.T, 1) / \
        np.expand_dims(pdf, [1, 2])
    nb_ranges = np.concatenate(([0], nb_ranges), axis=0)
    # sum (integration)
    weighted_latent_per_sample = \
        np.zeros([num_samples, num_features[0], hidden_size])
    for i in range(num_samples):
      weighted_latent_per_sample[i] = \
          np.sum(weighted_features_per_nb[nb_ranges[i]:nb_ranges[i + 1]],
                 axis=0)

    self.assertAllClose(weighted_latent_per_sample_tf,
                        weighted_latent_per_sample)


  @parameterized.parameters(
    (100, 4, [3, 3], 2, np.sqrt(3), 8, 3)
  )
  def test_basis_proj_jacobian(self,
                               num_points,
                               num_samples,
                               num_features,
                               batch_size,
                               radius,
                               hidden_size,
                               dimension):
    cell_sizes = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points, dimension=dimension)
    features = np.random.rand(num_points, num_features[0])
    point_cloud = PointCloud(points, batch_ids)

    point_samples, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples, dimension=dimension)

    point_cloud_samples = PointCloud(point_samples, batch_ids_samples)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    bandwidth = np.float32(np.repeat(0.2, dimension))
    pdf = neighborhood.get_pdf(bandwidth=bandwidth)
    nb_ids = neighborhood._original_neigh_ids
    # tf
    conv_layer = MCConv(
        num_features[0], num_features[1], dimension, hidden_size)

    neigh_point_coords = points[nb_ids[:, 0]]
    center_point_coords = point_samples[nb_ids[:, 1]]
    kernel_input = (neigh_point_coords - center_point_coords) / radius
    weights = conv_layer._basis_tf

    _, _, counts = tf.unique_with_counts(neighborhood._neighbors[:, 1])
    max_num_nb = tf.reduce_max(counts).numpy()

    with self.subTest(name='features'):
      def basis_proj_features(features_in):
        return basis_proj_tf(kernel_input,
                             neighborhood,
                             pdf,
                             features_in,
                             weights,
                             2) / (max_num_nb)
      self.assert_jacobian_is_correct_fn(
          basis_proj_features, [np.float32(features)], atol=1e-3)

    with self.subTest(name='weights'):
      def basis_proj_weights(weights_in):
        return basis_proj_tf(kernel_input,
                             neighborhood,
                             pdf,
                             features,
                             weights_in,
                             2) / (max_num_nb)
      self.assert_jacobian_is_correct_fn(
          basis_proj_weights, [weights], atol=1e-3)
    
    with self.subTest(name='points'):
      def basis_proj_points(points_in):
        neigh_point_coords = tf.gather(points_in, nb_ids[:, 0])
        center_point_coords = point_samples[nb_ids[:, 1]]
        kernel_input = (neigh_point_coords - center_point_coords) / radius
        return basis_proj_tf(kernel_input,
                             neighborhood,
                             pdf,
                             features,
                             weights,
                             2) / (max_num_nb)
      self.assert_jacobian_is_correct_fn(
          basis_proj_points, [np.float32(points)], atol=1e-3)


if __name__ == '__main___':
  test_case.main()
