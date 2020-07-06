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
""" Wrappers for point cloud CUDA functions """

import tensorflow as tf
import tfg_custom_ops


def compute_keys(point_cloud, aabb, num_cells, cell_size, name=None):
  with tf.compat.v1.name_scope(
      name, "compute keys", [point_cloud, aabb, num_cells, cell_size]):
    return tfg_custom_ops.compute_keys(
      point_cloud._points,
      point_cloud._batch_ids,
      aabb._aabb_min / cell_size,
      num_cells,
      tf.math.reciprocal(cell_size))
tf.no_gradient('ComputeKeys')


def build_grid_ds(keys, num_cells, batch_size, name=None):
  with tf.compat.v1.name_scope(
      name, "build grid ds", [keys, num_cells, batch_size]):
    return tfg_custom_ops.build_grid_ds(
      keys,
      num_cells,
      num_cells,
      batch_size)
tf.no_gradient('BuildGridDs')


def find_neighbors(grid, point_cloud_sampled, radii, max_neighbors, name=None):
  with tf.compat.v1.name_scope(
      name, "find neighbours",
      [grid, point_cloud_sampled, radii, max_neighbors]):
    return tfg_custom_ops.find_neighbors(
      point_cloud_sampled._points,
      point_cloud_sampled._batch_ids,
      grid._sorted_points,
      grid._sorted_keys,
      grid._fast_DS,
      grid._num_cells,
      grid._aabb._aabb_min / grid._cell_sizes,
      tf.math.reciprocal(grid._cell_sizes),
      tf.math.reciprocal(radii),
      max_neighbors)
tf.no_gradient('FindNeighbors')


def sampling(neighborhood, sample_mode, name=None):
  with tf.compat.v1.name_scope(name, "sampling", [neighborhood, sample_mode]):
    return tfg_custom_ops.sampling(
      neighborhood._grid._sorted_points,
      neighborhood._grid._sorted_batch_ids,
      neighborhood._grid._sorted_keys,
      neighborhood._grid._num_cells,
      neighborhood._neighbors,
      neighborhood._samples_neigh_ranges,
      sample_mode)
tf.no_gradient('sampling')


def compute_pdf(neighborhood, bandwidth, mode, name=None):
  with tf.compat.v1.name_scope(
      name, "compute pdf with point gradients",
      [neighborhood, bandwidth, mode]):
    return tfg_custom_ops.compute_pdf_with_pt_grads(
      neighborhood._grid._sorted_points,
      neighborhood._neighbors,
      neighborhood._samples_neigh_ranges,
      tf.math.reciprocal(bandwidth),
      tf.math.reciprocal(neighborhood._radii),
      mode)


@tf.RegisterGradient("ComputePdfWithPtGrads")
def _compute_pdf_grad(op, *grads):
  inPtsGrad = tfg_custom_ops.compute_pdf_pt_grads(
    op.inputs[0],
    op.inputs[1],
    op.inputs[2],
    op.inputs[3],
    op.inputs[4],
    grads[0],
    op.get_attr("mode"))
  return [inPtsGrad, None, None, None, None]


def basis_proj(kernel_inputs, neighborhood, features,
               basis, basis_type):
  pdf = neighborhood._pdf
  return tfg_custom_ops.basis_proj(
      kernel_inputs,
      features,
      neighborhood._original_neigh_ids,
      neighborhood._samples_neigh_ranges,
      pdf,
      basis,
      basis_type)


@tf.RegisterGradient("BasisProj")
def _basis_proj_grad(op, *grads):
  feature_grads, basis_grads, kernel_in_grads, pdf_grads = \
      tfg_custom_ops.basis_proj_grads(
          op.inputs[0], op.inputs[1], op.inputs[2],
          op.inputs[3], op.inputs[4], op.inputs[5],
          grads[0], op.get_attr("basis_type"))
  return [kernel_in_grads, feature_grads, None, None,
          pdf_grads, basis_grads]
