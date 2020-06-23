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
"""Class to test point clouds"""

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
from MCCNN2.pc.tests import utils


class point_cloud_test(test_case.TestCase):

  @parameterized.parameters(
    ([32], 100, 3),
    ([5, 2], 100, 2),
    ([2, 3, 4], 100, 4)
  )
  def test_flatten_unflatten_padded(self, batch_shape, num_points, dimension):
    batch_size = np.prod(batch_shape)
    points, sizes = utils._create_random_point_cloud_padded(
        num_points, batch_shape, dimension=dimension)
    point_cloud = PointCloud(points, sizes=sizes)
    retrieved_points = point_cloud.get_points().numpy()
    self.assertAllEqual(points.shape, retrieved_points.shape)
    points = points.reshape([batch_size, num_points, dimension])
    retrieved_points = retrieved_points.reshape(
        [batch_size, num_points, dimension])
    sizes = sizes.reshape([batch_size])
    for i in range(batch_size):
      self.assertAllClose(points[i, :sizes[i]], retrieved_points[i, :sizes[i]])
      self.assertTrue(np.all(retrieved_points[i, sizes[i]:] == 0))

  @parameterized.parameters(

  )
  def 

if __name__ == '__main__':
  test_case.main()
