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
from MCCNN2.pc.layers import MCConv2Sampled


class MCConvTest(test_case.TestCase):
  """
  @parameterized.parameters(
    # neighbor ids are currently corrupted on dimension 2: todo fix
    # (2000, 200, 16, 0.7, 2),
    # (4000, 400, 8, np.sqrt(2), 2),
    (2000, 200, [3, 3], 16, 0.7, 8, 3),
    # (4000, 400, 8, np.sqrt(3), 3),
    # (4000, 100, 1, np.sqrt(3), 3),
    # (2000, 200, 16, 0.7, 4),
    # (4000, 400, 8, np.sqrt(4), 4)
  )
  def test_convolution(self,
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
    # tf
    conv_layer = MCConv2Sampled(
        num_features[0], num_features[1], dimension, hidden_size)
    conv_result_tf = conv_layer(
        features, point_cloud, point_cloud_samples, radius, neighborhood)

    # numpy
    # sorted_points = grid._sorted_points.numpy()
    pdf = neighborhood.get_pdf().numpy()
    neighbor_ids = neighborhood._original_neigh_ids.numpy()
    nb_ranges = neighborhood._samples_neigh_ranges.numpy()
    hidden_weights = conv_layer._basis_tf[:, :-1].numpy()
    hidden_biases = conv_layer._basis_tf[:, -1].numpy()
    hidden_biases = np.expand_dims(hidden_biases, 1)
    weights = conv_layer._weights.numpy()
    weights = np.reshape(weights, [num_features[1], -1])

    features_on_neighbors = features[neighbor_ids[:, 0]]

    point_diff = (points[neighbor_ids[:, 0]] -\
                  point_samples[neighbor_ids[:, 1]])\
        / np.expand_dims(cell_sizes, 0)
    latent_per_nb = np.dot(hidden_weights, point_diff.T) + hidden_biases
    latent_relu_per_nb = np.maximum(latent_per_nb, 0)
    weighted_features_per_nb = np.expand_dims(features_on_neighbors, 2) * \
        np.expand_dims(latent_relu_per_nb.T, 1) / \
        np.expand_dims(pdf, [1, 2])
    nb_ranges = np.concatenate(([0], nb_ranges), axis=0)
    weighted_latent_per_sample = \
        np.zeros([num_samples, num_features[1], hidden_size])
    for i in range(num_samples):
      weighted_latent_per_sample[i] = \
          np.sum(weighted_features_per_nb[nb_ranges[i]:nb_ranges[i + 1]],
                 axis=0)
    weighted_latent_per_sample = np.reshape(weighted_latent_per_sample,
                                            [num_samples, -1])
    conv_result_np = np.dot(weighted_latent_per_sample, weights.T)

    self.assertAllClose(conv_result_tf, conv_result_np)

    # kernel_weights = np.dot(weights, latent.T)
    # kDiffs = (np.expand_dims(point_diff, 1) - np.expand_dims(mKernelPts[:, 0:3], 0))
    # lengths = np.sqrt(np.sum(kDiffs*kDiffs, axis=2))
    # #Linear
    # #correl = np.maximum(1.0 - lengths*2.0, 0.0)/np.expand_dims(pdfRes, 1)
    # #Exponential
    # correl = np.exp(-(lengths*lengths*2.0))/np.expand_dims(pdfRes, 1)
    # weightedFeat = np.expand_dims(sortedFeaturesRes[neighborRes[:,0]], 2)*np.expand_dims(correl, 1)
    # sampleIndicies = np.insert(sampleNeighsRes, 0, 0, axis=0)
    # sampleWeightedFeat = np.array([np.sum(weightedFeat[sampleIndicies[i]:sampleIndicies[i+1], :], axis=0) \
    #     if ((sampleIndicies[i+1]-sampleIndicies[i]) > 0) else np.zeros_like(weightedFeat[0]) for i in range(len(sampleNeighsRes))])
    # realOutFeatures = np.matmul(sampleWeightedFeat.reshape((-1, numInFeatures*numKernelPts)), mWeights)
  """

  @parameterized.parameters(
    # neighbor ids are currently corrupted on dimension 2: todo fix
    # (2000, 200, 16, 0.7, 2),
    # (4000, 400, 8, np.sqrt(2), 2),
    (2000, 200, [3, 3], 16, 0.7, 8, 3),
    # (4000, 400, 8, np.sqrt(3), 3),
    # (4000, 100, 1, np.sqrt(3), 3),
    # (2000, 200, 16, 0.7, 4),
    # (4000, 400, 8, np.sqrt(4), 4)
  )
  def test_conv_jacobian_params(self,
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
    features = np.tile(batch_ids, [num_features[0],1]).T
    point_cloud = PointCloud(points, batch_ids)

    point_samples, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples, dimension=dimension)

    point_cloud_samples = PointCloud(point_samples, batch_ids_samples)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    conv_layer = MCConv2Sampled(
        num_features[0], num_features[1], dimension, hidden_size)

    with self.subTest(name='params_basis_proj'):
      def conv_basis(basis_tf_in):
        conv_layer._basis_tf = basis_tf_in
        conv_result = conv_layer(
          features, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      basis_tf = conv_layer._basis_tf
      self.assert_jacobian_is_correct_fn(
          conv_basis, [basis_tf], atol=1e-4)
    
    with self.subTest(name='params_second_layer'):
      def conv_weights(weigths_in):
        conv_layer._weights = weigths_in
        conv_result = conv_layer(
          features, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      weights = conv_layer._weights
      self.assert_jacobian_is_correct_fn(
          conv_weights, [weights], atol=1e-4)

  @parameterized.parameters(
    # neighbor ids are currently corrupted on dimension 2: todo fix
    # (2000, 200, 16, 0.7, 2),
    # (4000, 400, 8, np.sqrt(2), 2),
    (200, 20, [3, 3], 16, np.sqrt(3), 8, 3),
    # (4000, 400, 8, np.sqrt(3), 3),
    # (4000, 100, 1, np.sqrt(3), 3),
    # (2000, 200, 16, 0.7, 4),
    # (4000, 400, 8, np.sqrt(4), 4)
  )
  def test_conv_jacobian_points(self,
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

    point_samples, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples, dimension=dimension)
    point_cloud_samples = PointCloud(point_samples, batch_ids_samples)
    point_cloud = PointCloud(points, batch_ids)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)

    def conv_points(points_in):
      point_cloud = PointCloud(points_in, batch_ids)
      # neighborhood._grid._sorted_points = \
      #     tf.gather(
      #       points_in, grid._sorted_indices)
      conv_layer = MCConv2Sampled(
          num_features[0], num_features[1], dimension, hidden_size)
      conv_result = conv_layer(
        features, point_cloud, point_cloud_samples, radius, neighborhood)
      return conv_result

    self.assert_jacobian_is_correct_fn(
        conv_points, [np.float32(points)], atol=1e-4)


if __name__ == '__main___':
  test_case.main()
