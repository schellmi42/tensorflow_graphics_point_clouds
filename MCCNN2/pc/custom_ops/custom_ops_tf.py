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
""" Non-GPU implemetations of the custom ops """

import tensorflow as tf
from MCCNN2.pc import PointCloud


def compute_keys_tf(point_cloud: PointCloud, num_cells, cell_size, name=None):
  """
    For a point in cell `c` the key is computed as
        \\(key = batch_id * prod_{d=0}^{D} num_cells_{d} + \\)
        \\(sum_{d=0}^{D}( c_{d} prod_{d'=d+1}^{D} num_cells_{d'} ) \\).
    Args:
      point_cloud: A `PointCloud` instance.
      num_cells: An `int` tensor of shape [D], the total number of cells
        per dimension.
      cell_size: An `int` tensor of shape [D], the cell sizes per dimension.

    Returns:
      An `int` tensor of shape [N], the keys per point.
  """
  with tf.compat.v1.name_scope(
      name, "compute keys", [point_cloud, num_cells, cell_size]):
    aabb = point_cloud.get_AABB()
    abb_min_per_batch = aabb._aabb_min
    aabb_min_per_point = tf.gather(abb_min_per_batch, point_cloud._batch_ids)
    cell_ind = tf.math.floor(
        (point_cloud._points - aabb_min_per_point) / cell_size)
    cell_ind = tf.cast(cell_ind, tf.int32)
    cell_ind = tf.minimum(
        tf.maximum(cell_ind, tf.zeros_like(cell_ind)),
        num_cells)
    cell_multiplier = tf.math.cumprod(num_cells, reverse=True)
    cell_multiplier = tf.concat((cell_multiplier, [1]), axis=0)
    keys = point_cloud._batch_ids * cell_multiplier[0] + \
        tf.math.reduce_sum(cell_ind * tf.reshape(cell_multiplier[1:], [1, -1]),
                           axis=1)
    return tf.cast(keys, tf.int64)


def build_grid_ds_tf(keys, num_cells, batch_size, name=None):
  """ Method to build a fast access data structure for point clouds.

  Creates a 2D regular grid in the first two dimension, saving the first and
  last index belonging to that cell array.
  Args:
    sorted_keys: An `int` tensor of shape [N], the sorted keys.
    num_cells: An `int` tensor of shape [D], the total number of cells
      per dimension.
    batch_size: An `int`.

  Returns:
    An `int` tensor of shape [batch_size, num_cells[0], num_cells[1], 2].
  """
  with tf.compat.v1.name_scope(
    name, 'build_grid_ds', [keys, num_cells, batch_size]):

    keys = tf.cast(tf.convert_to_tensor(value=keys), tf.int32)
    num_cells = tf.cast(tf.convert_to_tensor(value=num_cells), tf.int32)

    num_keys = keys.shape[0]
    num_cells_2D = batch_size * num_cells[0] * num_cells[1]

    if num_cells.shape[0] > 2:
        cells_per_2D_cell = tf.reduce_prod(num_cells[2:])
    elif num_cells.shape[0] == 2:
        cells_per_2D_cell = 1

    ds_indices = tf.cast(tf.floor(keys / cells_per_2D_cell), dtype=tf.int32)
    # y_indices = tf.math.mod(ds_indices, num_cells[1])
    # y_remainder = tf.floor(ds_indices / num_cells[1])
    # x_index = tf.math.mod(y_remainder, num_cells[0])
    # batch_ids = tf.floor(y_remainder / num_cells[0])

    indices = tf.range(0, num_keys, dtype=tf.int32)

    first_per_cell = tf.math.unsorted_segment_min(
        indices, ds_indices, num_cells_2D)
    last_per_cell = tf.math.unsorted_segment_max(
        indices + 1, ds_indices, num_cells_2D)

    empty_cells = first_per_cell < 0
    first_per_cell = tf.where(
        empty_cells, tf.zeros_like(first_per_cell), first_per_cell)
    last_per_cell = tf.where(
        empty_cells, tf.zeros_like(last_per_cell), last_per_cell)
    empty_cells = first_per_cell > num_keys
    first_per_cell = tf.where(
        empty_cells, tf.zeros_like(first_per_cell), first_per_cell)
    last_per_cell = tf.where(
        empty_cells, tf.zeros_like(last_per_cell), last_per_cell)
    # cell_change = ds_indices[1:] != ds_indices[:-1]

    # last_per_cell = tf.boolean_mask(tf.range(0, num_keys - 1), cell_change)
    # first_per_cell = last_per_cell + 1
    # last_per_cell = tf.concat((last_per_cell, [num_keys]), axis=0)
    # first_per_cell = tf.concat(([0], first_per_cell), axis=0)

    return tf.stack([tf.reshape(first_per_cell,
                                [batch_size, num_cells[0], num_cells[1]]),
                     tf.reshape(last_per_cell,
                                [batch_size, num_cells[0], num_cells[1]])],
                    axis=3)


# def build_grid_ds(pKeys, pNumCells, pbatch_size, name=None):
#   with tf.compat.v1.name_scope(
#       name, "build grid ds", [pKeys, pNumCells, pbatch_size]):
#     return tfg_custom_ops.build_grid_ds(
#       pKeys,
#       pNumCells,
#       pNumCells,
#       pbatch_size)
# tf.no_gradient('BuildGridDs')


# def find_neighbors(pGrid, pPCSamples, pRadii, pMaxNeighbors, name=None):
#   with tf.compat.v1.name_scope(
#       name, "find neighbours", [pGrid, pPCSamples, pRadii, pMaxNeighbors]):
#     return tfg_custom_ops.find_neighbors(
#       pPCSamples._points,
#       pPCSamples._batch_ids,
#       pGrid.sorted_points,
#       pGrid.sortedKeys_,
#       pGrid.fastDS_,
#       pGrid.numCells_,
#       pGrid.aabb_.aabbMin_ / pGrid.cellSizes_,
#       tf.math.reciprocal(pGrid.cellSizes_),
#       tf.math.reciprocal(pRadii),
#       pMaxNeighbors)
# tf.no_gradient('FindNeighbors')


# def sampling(pNeighborhood, pSampleMode, name=None):
#   with tf.compat.v1.name_scope(name, "sampling",
#       [pNeighborhood, pSampleMode]):
#     return tfg_custom_ops.sampling(
#       pNeighborhood.grid_.sorted_points,
#       pNeighborhood.grid_.sorted_batch_ids,
#       pNeighborhood.grid_.sortedKeys_,
#       pNeighborhood.grid_.numCells_,
#       pNeighborhood.neighbors_,
#       pNeighborhood.samplesNeighRanges_,
#       pSampleMode)
# tf.no_gradient('sampling')


# def compute_pdf(pNeighborhood, pBandwidth, pMode, name=None):
#   with tf.compat.v1.name_scope(
#       name, "compute pdf with point gradients",
#       [pNeighborhood, pBandwidth, pMode]):
#     return tfg_custom_ops.compute_pdf_with_pt_grads(
#       pNeighborhood.grid_.sorted_points,
#       pNeighborhood.neighbors_,
#       pNeighborhood.samplesNeighRanges_,
#       tf.math.reciprocal(pBandwidth),
#       tf.math.reciprocal(pNeighborhood.radii_),
#       pMode)


# @tf.RegisterGradient("ComputePdfWithPtGrads")
# def _compute_pdf_grad(op, *grads):
#   inPtsGrad = tfg_custom_ops.compute_pdf_pt_grads(
#     op.inputs[0],
#     op.inputs[1],
#     op.inputs[2],
#     op.inputs[3],
#     op.inputs[4],
#     grads[0],
#     op.get_attr("mode"))
#   return [inPtsGrad, None, None, None, None]


# def basis_proj(pKernelInputs, pNeighborhood, pInFeatures,
#                pBasis, pBasisType):
#   curPDF = pNeighborhood.pdf_
#   return tfg_custom_ops.basis_proj(
#       pKernelInputs,
#       pInFeatures,
#       pNeighborhood.originalNeighIds_,
#       pNeighborhood.samplesNeighRanges_,
#       curPDF,
#       pBasis,
#       pBasisType)


# @tf.RegisterGradient("BasisProj")
# def _basis_proj_grad(op, *grads):
#   featGrads, basisGrads, kernelInGrads, pdfGrads = \
#       tfg_custom_ops.basis_proj_grads(
#           op.inputs[0], op.inputs[1], op.inputs[2],
#           op.inputs[3], op.inputs[4], op.inputs[5],
#           grads[0], op.get_attr("basis_type"))
#   return [kernelInGrads, featGrads, None, None,
#           pdfGrads, basisGrads]
