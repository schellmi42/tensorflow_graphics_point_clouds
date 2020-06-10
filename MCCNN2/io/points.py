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



# TODO(b/156115314): Revisit the library for loading the triangle meshes.
class GFileResolver(trimesh.visual.resolvers.Resolver):
  """A resolver using gfile for accessing other assets in the mesh directory."""

  def __init__(self, path):
    if tf.io.gfile.isdir(path):
      self.directory = path
    elif tf.io.gfile.exists(path):
      self.directory = os.path.dirname(path)
    else:
      raise ValueError('path is not a file or directory')

  def get(self, name):
    with tf.io.gfile.GFile(os.path.join(self.directory, name), 'rb') as f:
      data = f.read()
    return data


def load(file_obj, file_type=None, **kwargs):
  """Loads a triangle mesh from the given GFile/file path.

  Args:
    file_obj: A tf.io.gfile.GFile object or a string specifying the mesh file
      path.
    file_type: A string specifying the type of the file (e.g. 'obj', 'stl'). If
      not specified the file_type will be inferred from the file name.
    **kwargs: Additional arguments that should be passed to trimesh.load().

  Returns:
    A trimesh.Trimesh or trimesh.Scene.
  """

  if isinstance(file_obj, str):
    with tf.io.gfile.GFile(file_obj, 'r') as f:
      if file_type is None:
        file_type = trimesh.util.split_extension(file_obj)
      return trimesh.load(
          file_obj=f,
          file_type=file_type,
          resolver=GFileResolver(file_obj),
          **kwargs)

  if trimesh.util.is_file(file_obj):
    if not hasattr(file_obj, 'name') or not file_obj.name:
      raise ValueError(
          'file_obj must have attribute "name". Try passing the file name instead.'
      )
    if file_type is None:
      file_type = trimesh.util.split_extension(file_obj.name)
    return trimesh.load(
        file_obj=file_obj,
        file_type=file_type,
        resolver=GFileResolver(file_obj.name),
        **kwargs)

  raise ValueError('file_obj should be either a file object or a string')


__all__ = ['load', 'Trimesh', 'Scene']
