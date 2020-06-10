# Copyright 2020 Google LLC
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
# Lint as: python3
"""Small functions to load point clouds"""

import os
import tensorflow as tf
import trimesh
from trimesh import Scene
from trimesh import Trimesh


def load_points_from_file(filename, delimiter=',', dimension = 3, dtype=tf.float32, name=None):

    points = []
    features =[]
    with open(filename,'r') as in_file:
      for line in in_file:
        line_elements = line[:-1].split(delimiter)
        points.append(line_elements[0:dimension])
        if len(line_elements)>3:
          features.append(line_elements[dimension:])
    with tf.compat.v1.name_scope(name, "point loader", [points, features, dtype]):
      points = tf.convert_to_tensor(value=points)
      features = tf.convert_to_tensor(value=features)
      points = tf.strings.to_number(input=points, out_type=dtype)
      features = tf.strings.to_number(input=features, out_type=dtype)
    return points, features

def load_batch_of_points(filenames, batch_shape=[-1], delimiter=',', point_dimension=3, dtype=tf.float32, name=None):
    with tf.compat.v1.name_scope(name, "load batch of points", [filenames, batch_shape, delimiter, point_dimension, dtype]):
      batch_size = len(filenames)
      if tf.reduce_prod(batch_shape)!=batch_size:
        raise ValueError('Invalid batch shape %s for batch size %s'%(batch_shape, batch_size))
      points = []
      features = []
      max_num_points = 0
      sizes = []
      for filename in filenames:
        curr_points, curr_features = load_points_from_file(filename=filename, delimiter=delimiter, dtype=dtype)
        points.append(curr_points)
        features.append(curr_features)
        sizes.append(len(curr_points))
      
      sizes = tf.convert_to_tensor(value=sizes)
      max_num_points = tf.reduce_max(sizes)
      

      feature_dimension = features[0].shape[1]
      for i in range(batch_size):
        pad_size = max_num_points - sizes[i]
        points[i] = tf.concat((points[i], tf.zeros(shape=[pad_size,point_dimension], dtype=dtype)),axis=0)
        features[i] = tf.concat((features[i], tf.zeros(shape=[pad_size, feature_dimension], dtype=dtype)),axis=0)
      
      points = tf.stack(values=points, axis=0)
      features = tf.stack(values=features, axis=0)
      
      points = tf.reshape(tensor=points, shape=batch_shape+[max_num_points, point_dimension])
      features = tf.reshape(tensor=features, shape=batch_shape+[max_num_points, feature_dimension])
      sizes = tf.reshape(tensor=sizes, shape=batch_shape)

      return points, features, sizes
