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
    aabbMin_: A float 'Tensor' of shape [B,D], list of minimum points of the
      bounding boxes.
    aabbMax_: A float 'Tensor' of shape [B,D], list of maximum points of the
      bounding boxes.
    batchSize_: An integer, size of the batch.
    batchShape_: An int 'Tensor' of shape [B], the batch shape [A1,...,An]
  """

  def __init__(self, point_cloud, name=None):
    """Constructor.

    Args:
      Pointcloud: A 'PointCloud' instance from which to compute the
        bounding box.
    """
    with tf.compat.v1.name_scope(
        name, "bounding box constructor", [self, point_cloud]):
      self.batchSize_ = point_cloud.batchSize_
      self.batchShape_ = point_cloud.batchShape_
      self.point_cloud_ = point_cloud

      self.aabbMin_ = tf.math.unsorted_segment_min(
          data=point_cloud.pts_, segment_ids=point_cloud.batchIds_,
          num_segments=self.batchSize_) - 1e-9
      self.aabbMax_ = tf.math.unsorted_segment_max(
          data=point_cloud.pts_, segment_ids=point_cloud.batchIds_,
          num_segments=self.batchSize_) + 1e-9

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
      diam = tf.linalg.norm(self.aabbMax_ - self.aabbMin_, ord=ord, axis=-1)
      if self.batchShape_ is None:
        return diam
      else:
        return tf.reshape(diam, self.batchShape_)


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
    pts_ (float tensor [N,D]): List of points.
    sizes_ (int tensor [A1,,.An]): sizes of the point clouds, None if
      constructed using segmented input
    batchIds_ (int tensor n): List of batch ids associated with the points.
    batchSize_ (int): Size of the batch.
    batchShape_ (int tensor): [A1,..,An] original shape of the batch, None
      if constructed using segmented input
    unflatten_ (function): Function to reshape segmented [N,D] to
      [A1,...,An,V,D], zero padded, None if constructed using
      segmented input
    dimension_ (int): dimensionality of the point clouds
    get_segment_id_ (int tensor [A1,...,An]): tensor that returns the
      segment id given a relative id in [A1,...,An]
    aabb_: A `AABB` instance, the bounding box of the point cloud.
  """

  def __init__(self,
               pPts,
               pBatchIds=None,
               pBatchSize=None,
               sizes=None,
               name=None):
    """Constructor.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      pPts: A float tensor either of shape [N,D]
        or of shape [A1,..,An,V,D], possibly padded as indicated by sizes.
        Represents the point coordinates.
      pBatchIds: An int tensor of shape [N] associated with the points of pPts
      sizes:      An `int` tensor of shape `[A1, ..., An]` indicating the
        true input sizes in case of padding (`sizes=None` indicates no padding)
        Note that `sizes[A1, ..., An] <= V` or `sum(sizes) == N`.
      pBatchSize (int): Size of the batch.
    """
    with tf.compat.v1.name_scope(
        name, "construct point cloud",
        [self, pPts, pBatchIds, pBatchSize, sizes]):
      pPts = tf.convert_to_tensor(value=pPts, dtype=tf.float32)
      if sizes is not None:
        sizes = tf.convert_to_tensor(value=sizes, dtype=tf.int32)
      if pBatchIds is not None:
        pBatchIds = tf.convert_to_tensor(value=pBatchIds, dtype=tf.int32)

      check_valid_point_cloud_input(pPts, sizes, pBatchIds)

      self.sizes_ = sizes
      self.batchSize_ = pBatchSize
      self.batchIds_ = pBatchIds
      self.dimension_ = pPts.shape[-1]
      self.batchShape_ = None
      self.unflatten_ = None
      self.aabb_ = None

      if len(pPts.shape) > 2:
        self._init_from_padded(pPts)
      else:
        self._init_from_segmented(pPts)

      #Sort the points based on the batch ids in incremental order.
      self.sortedIndicesBatch_ = tf.argsort(self.batchIds_)

      # initialize grid and neighborhood_cache
      self._grid_cache = {}
      self._neighborhood_cache = {}

  def _init_from_padded(self, points):
    """converting padded [A1,...,An,V,D] tensor into a 2D tensor [N,D] with
    segmentation ids
    """
    self.batchShape_ = points.shape[:-2]
    if self.batchSize_ is None:
      self.batchSize_ = tf.reduce_prod(self.batchShape_)
    if self.sizes_ is None:
      self.sizes_ = tf.constant(
          value=points.shape[-2], shape=self.batchShape_)
    self.get_segment_id_ = tf.reshape(
        tf.range(0, self.batchSize_), self.batchShape_)
    self.pts_, self.unflatten_ = flatten_batch_to_2d(points, self.sizes_)
    self.batchIds_ = tf.repeat(
        tf.range(0, self.batchSize_),
        repeats=tf.reshape(self.sizes_, [-1]))

  def _init_from_segmented(self, points):
    """if input is already 2D tensor with segmentation ids or given sizes
    """
    if self.batchIds_ is None:
      if self.batchSize_ is None:
        self.batchSize_ = tf.reduce_prod(self.sizes_.shape)
      self.batchIds_ = tf.repeat(tf.range(0, self.batchSize_), self.sizes_)
    if self.batchSize_ is None:
      self.batchSize_ = tf.reduce_max(self.batchIds_) + 1
    self.pts_ = points

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
          slice = self.get_segment_id_
          for slice_id in id:
            slice = slice[slice_id]
          id = slice
        if id > self.batchSize_:
          raise IndexError('batch index out of range')
        return self.pts_[self.batchIds_ == id]
      else:
        return self.get_unflatten(max_num_points=max_num_points)(self.pts_)

  def get_sizes(self, name=None):
    """ Returns the sizes of the point clouds in the batch.

    Note:
      In the following, A1 to An are optional batch dimensions.
      Use this instead of accessing 'self.sizes_',
      if the class was constructed using segmented input the 'sizes_' is
      created in this method.

    Return:
      tensor of shape [A1,..,An]
    """
    with tf.compat.v1.name_scope(name, "get point cloud sizes", [self]):
      if self.sizes_ is None:
        auxIdx, _, self.sizes_ = tf.unique_with_counts(
            self.batchIds_)
        auxIdxSortedIds = tf.argsort(auxIdx)
        self.sizes_ = tf.gather(self.sizes_, auxIdxSortedIds)
        if self.batchShape_ is not None:
          self.sizes_ = tf.reshape(self.sizes_, self.batchShape_)
      return self.sizes_

  def get_unflatten(self, max_num_points=None, name=None):
    """ Returns the method to unflatten the segmented points.

    Use this instead of accessing 'self.unflatten_',
    if the class was constructed using segmented input the 'unflatten_' method
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
      if self.unflatten_ is None:
        self.unflatten_ = lambda data: unflatten_2d_to_batch(
            data=tf.gather(data, self.sortedIndicesBatch_),
            sizes=self.get_sizes(),
            max_rows=max_num_points)
      return self.unflatten_

  def get_AABB(self) -> _AABB:
    """ Returns the axis aligned bounding box of the point cloud.

    Use this instead of accessing `self.aabb_`, as the bounding box
    is initialized  with tthe first call of his method.

    Returns:
      A `AABB` instance
    """
    if self.aabb_ is None:
      self.aabb_ = _AABB(point_cloud=self)
    return self.aabb_

  def set_batch_shape(self, batchShape, name=None):
    """ Function to change the batch shape

      Use this to set a batch shape instead of using 'self.batchShape_' to also
      change dependent variables.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      batchShape: float tensor of shape [A1,...,An]

    Raises:
      ValueError: if shape does not sum up to batch size.
    """
    with tf.compat.v1.name_scope(
        name, "set batch shape of point cloud", [self, batchShape]):
      if batchShape is not None:
        batchShape = tf.convert_to_tensor(value=batchShape, dtype=tf.int32)
        if tf.reduce_prod(batchShape) != self.batchSize_:
          raise ValueError(
              f'Incompatible batch size. Must be {self.batchSize_} \
               but is {tf.reduce_prod(batchShape)}')
        self.batchShape_ = batchShape
        self.get_segment_id_ = tf.reshape(
            tf.range(0, self.batchSize_), self.batchShape_)
        if self.sizes_ is not None:
          self.sizes_ = tf.reshape(self.sizes_, self.batchShape_)
      else:
        self.batchShape_ = None

  def __eq__(self, other, name=None):
    """Comparison operator.

    Args:
      other (PointCloud): Other point cloud.
    Return:
      True if other is equal to self, False otherwise.
    """
    with tf.compat.v1.name_scope(name, "compare point clouds", [self, other]):
      return self.pts_.name == other.pts_.name and \
        self.batchIds_.name == other.batchIds_.name and \
        self.batchSize_ == other.batchSize_

  def __hash__(self, name=None):
    """Method to compute the hash of the object.

    Return:
      Unique hash value.
    """
    with tf.compat.v1.name_scope(name, "hash point cloud", [self]):
      return hash((self.pts_.name, self.batchIds_.name, self.batchSize_))
