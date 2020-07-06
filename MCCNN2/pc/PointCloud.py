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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class to represent point clouds"""

import tensorflow as tf
from tensorflow_graphics.geometry.convolution.utils import \
    flatten_batch_to_2d, unflatten_2d_to_batch

from MCCNN2.pc.utils import check_valid_point_cloud_input


class _AABB:
  """Class to represent axis aligned bounding box of point clouds.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Attributes:
    _aabb_min: A float 'Tensor' of shape [B,D], list of minimum points of the
      bounding boxes.
    _aabb_max: A float 'Tensor' of shape [B,D], list of maximum points of the
      bounding boxes.
    _batch_size: An integer, size of the batch.
    _batch_shape: An int 'Tensor' of shape [B], the batch shape [A1,...,An]
  """

  def __init__(self, point_cloud, name=None):
    """Constructor.

    Args:
      Pointcloud: A 'PointCloud' instance from which to compute the
        bounding box.
    """
    with tf.compat.v1.name_scope(
        name, "bounding box constructor", [self, point_cloud]):
      self._batch_size = point_cloud._batch_size
      self._batch_shape = point_cloud._batch_shape
      self.point_cloud_ = point_cloud

      self._aabb_min = tf.math.unsorted_segment_min(
          data=point_cloud._points, segment_ids=point_cloud._batch_ids,
          num_segments=self._batch_size) - 1e-9
      self._aabb_max = tf.math.unsorted_segment_max(
          data=point_cloud._points, segment_ids=point_cloud._batch_ids,
          num_segments=self._batch_size) + 1e-9

  def get_diameter(self, ord='euclidean', name=None):
    """ Returns the diameter of the bounding box.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      ord:    Order of the norm. Supported values are `'euclidean'`,
          `1`, `2`, `np.inf` and any positive real number yielding the
          corresponding p-norm. Default is `'euclidean'`.
    Return:
      diam: A float 'Tensor' of shape [A1,..An], diameters of the
        bounding boxes
    """

    with tf.compat.v1.name_scope(
        name, "Compute diameter of bounding box",
        [self, ord]):
      diam = tf.linalg.norm(self._aabb_max - self._aabb_min, ord=ord, axis=-1)
      if self._batch_shape is None:
        return diam
      else:
        return tf.reshape(diam, self._batch_shape)


class PointCloud:
  """Class to represent a point cloud.

  Note:
    In the following, A1 to An are optional batch dimensions.

  The internal representation is a 2D tensor of shape [N,D] with
  segmentation ids of shape [N]. Can be constructed by directly passing
  segmented inputs or by prividing a [A1,..,An,V,D] padded tensor with
  `sizes` indicating the first elements from V to select from each batch
  dimension.

  Attributes:
    _points (float tensor [N,D]): List of points.
    _sizes (int tensor [A1,,.An]): sizes of the point clouds, None if
      constructed using segmented input
    _batch_ids (int tensor n): List of batch ids associated with the points.
    _batch_size (int): Size of the batch.
    _batch_shape (int tensor): [A1,..,An] original shape of the batch, None
      if constructed using segmented input
    _unflatten (function): Function to reshape segmented [N,D] to
      [A1,...,An,V,D], zero padded, None if constructed using
      segmented input
    _dimension (int): dimensionality of the point clouds
    _get_segment_id (int tensor [A1,...,An]): tensor that returns the
      segment id given a relative id in [A1,...,An]
    _aabb: A `AABB` instance, the bounding box of the point cloud.
  """

  def __init__(self,
               points,
               batch_ids=None,
               batch_size=None,
               sizes=None,
               name=None):
    """Constructor.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      points: A float tensor either of shape [N,D]
        or of shape [A1,..,An,V,D], possibly padded as indicated by sizes.
        Represents the point coordinates.
      batch_ids: An int tensor of shape [N] associated with the points.
      sizes:      An `int` tensor of shape `[A1, ..., An]` indicating the
        true input sizes in case of padding (`sizes=None` indicates no padding)
        Note that `sizes[A1, ..., An] <= V` or `sum(sizes) == N`.
      batch_size (int): Size of the batch.
    """
    with tf.compat.v1.name_scope(
        name, "construct point cloud",
        [self, points, batch_ids, batch_size, sizes]):
      points = tf.convert_to_tensor(value=points, dtype=tf.float32)
      if sizes is not None:
        sizes = tf.convert_to_tensor(value=sizes, dtype=tf.int32)
      if batch_ids is not None:
        batch_ids = tf.convert_to_tensor(value=batch_ids, dtype=tf.int32)

      check_valid_point_cloud_input(points, sizes, batch_ids)

      self._sizes = sizes
      self._batch_size = batch_size
      self._batch_ids = batch_ids
      self._dimension = points.shape[-1]
      self._batch_shape = None
      self._unflatten = None
      self._aabb = None

      if len(points.shape) > 2:
        self._init_from_padded(points)
      else:
        self._init_from_segmented(points)

      #Sort the points based on the batch ids in incremental order.
      self._sorted_indices_batch = tf.argsort(self._batch_ids)

      # initialize grid and neighborhood_cache
      self._grid_cache = {}
      self._neighborhood_cache = {}

  def _init_from_padded(self, points):
    """converting padded [A1,...,An,V,D] tensor into a 2D tensor [N,D] with
    segmentation ids
    """
    self._batch_shape = points.shape[:-2]
    if self._batch_size is None:
      self._batch_size = tf.reduce_prod(self._batch_shape)
    if self._sizes is None:
      self._sizes = tf.constant(
          value=points.shape[-2], shape=self._batch_shape)
    self._get_segment_id = tf.reshape(
        tf.range(0, self._batch_size), self._batch_shape)
    self._points, self._unflatten = flatten_batch_to_2d(points, self._sizes)
    self._batch_ids = tf.repeat(
        tf.range(0, self._batch_size),
        repeats=tf.reshape(self._sizes, [-1]))

  def _init_from_segmented(self, points):
    """if input is already 2D tensor with segmentation ids or given sizes
    """
    if self._batch_ids is None:
      if self._batch_size is None:
        self._batch_size = tf.reduce_prod(self._sizes.shape)
      self._batch_ids = tf.repeat(tf.range(0, self._batch_size), self._sizes)
    if self._batch_size is None:
      self._batch_size = tf.reduce_max(self._batch_ids) + 1
    self._points = points

  def get_points(self, id=None, max_num_points=None, name=None):
    """ Returns the points.

    Note:
      In the following, A1 to An are optional batch dimensions.

      If called withoud specifying 'id' returns the points in padded format
      [A1,...,An,V,D]

    Args:
      id (int): Identifier of point cloud in the batch, if None return all
      max_num_points: (int) specifies the 'V' dimension the method returns,
          by default uses maximum of 'sizes'. `max_rows >= max(sizes)`

    Return:
      tensor:  if 'id' was given: 2D float tensor of shape
        if 'id' not given: float tensor of shape [A1,...,An,V,D], zero padded
    """
    with tf.compat.v1.name_scope(
        name, "get point clouds", [self, id, max_num_points]):
      if id is not None:
        if not isinstance(id, int):
          slice = self._get_segment_id
          for slice_id in id:
            slice = slice[slice_id]
          id = slice
        if id > self._batch_size:
          raise IndexError('batch index out of range')
        return self._points[self._batch_ids == id]
      else:
        return self.get_unflatten(max_num_points=max_num_points)(self._points)

  def get_sizes(self, name=None):
    """ Returns the sizes of the point clouds in the batch.

    Note:
      In the following, A1 to An are optional batch dimensions.
      Use this instead of accessing 'self._sizes',
      if the class was constructed using segmented input the '_sizes' is
      created in this method.

    Return:
      tensor of shape [A1,..,An]
    """
    with tf.compat.v1.name_scope(name, "get point cloud sizes", [self]):
      if self._sizes is None:
        _ids, _, self._sizes = tf.unique_with_counts(
            self._batch_ids)
        _ids_sorted = tf.argsort(_ids)
        self._sizes = tf.gather(self._sizes, _ids_sorted)
        if self._batch_shape is not None:
          self._sizes = tf.reshape(self._sizes, self._batch_shape)
      return self._sizes

  def get_unflatten(self, max_num_points=None, name=None):
    """ Returns the method to unflatten the segmented points.

    Use this instead of accessing 'self._unflatten',
    if the class was constructed using segmented input the '_unflatten' method
    is created in this method.

    Note:
      In the following, A1 to An are optional batch dimensions

    Args:
      max_num_points: (int) specifies the 'V' dimension the method returns,
        by default uses maximum of 'sizes'. `max_rows >= max(sizes)`
    Return:
      method to unflatten the segmented points, which returns [A1,...,An,V,D]
      tensor, zero padded

    Raises:
      ValueError: When trying to unflatten unsorted points.
    """
    with tf.compat.v1.name_scope(
        name, "get unflatten method", [self, max_num_points]):
      if self._unflatten is None:
        self._unflatten = lambda data: unflatten_2d_to_batch(
            data=tf.gather(data, self._sorted_indices_batch),
            sizes=self.get_sizes(),
            max_rows=max_num_points)
      return self._unflatten

  def get_AABB(self) -> _AABB:
    """ Returns the axis aligned bounding box of the point cloud.

    Use this instead of accessing `self._aabb`, as the bounding box
    is initialized  with tthe first call of his method.

    Returns:
      A `AABB` instance
    """
    if self._aabb is None:
      self._aabb = _AABB(point_cloud=self)
    return self._aabb

  def set_batch_shape(self, batch_shape, name=None):
    """ Function to change the batch shape

      Use this to set a batch shape instead of using 'self._batch_shape' to
      also change dependent variables.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      batch_shape: float tensor of shape [A1,...,An]

    Raises:
      ValueError: if shape does not sum up to batch size.
    """
    with tf.compat.v1.name_scope(
        name, "set batch shape of point cloud", [self, batch_shape]):
      if batch_shape is not None:
        batch_shape = tf.convert_to_tensor(value=batch_shape, dtype=tf.int32)
        if tf.reduce_prod(batch_shape) != self._batch_size:
          raise ValueError(
              f'Incompatible batch size. Must be {self._batch_size} \
               but is {tf.reduce_prod(batch_shape)}')
        self._batch_shape = batch_shape
        self._get_segment_id = tf.reshape(
            tf.range(0, self._batch_size), self._batch_shape)
        if self._sizes is not None:
          self._sizes = tf.reshape(self._sizes, self._batch_shape)
      else:
        self._batch_shape = None

  def __eq__(self, other, name=None):
    """Comparison operator.

    Args:
      other (PointCloud): Other point cloud.
    Return:
      True if other is equal to self, False otherwise.
    """
    with tf.compat.v1.name_scope(name, "compare point clouds", [self, other]):
      return self._points.name == other._points.name and \
        self._batch_ids.name == other._batch_ids.name and \
        self._batch_size == other._batch_size

  def __hash__(self, name=None):
    """Method to compute the hash of the object.

    Return:
      Unique hash value.
    """
    with tf.compat.v1.name_scope(name, "hash point cloud", [self]):
      return hash((self._points.name, self._batch_ids.name, self._batch_size))
