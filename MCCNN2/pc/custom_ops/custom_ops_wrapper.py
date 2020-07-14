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


def compute_keys(point_cloud, num_cells, cell_size, name=None):
  """ Method to compute the regular grid keys of a point cloud.

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
    return tfg_custom_ops.compute_keys(
      point_cloud._points,
      point_cloud._batch_ids,
      aabb._aabb_min / cell_size,
      num_cells,
      tf.math.reciprocal(cell_size))
tf.no_gradient('ComputeKeys')


def build_grid_ds(sorted_keys, num_cells, batch_size, name=None):
  """ Method to build a fast access data structure for point clouds.

  Creates a 2D regular grid in the first two dimension, saving the first and
  last index belonging to that cell array.
  Args:
    sorted_keys: An `int` tensor of shape [N], the sorted keys.
    num_cells: An `int` tensor of shape [D], the total number of cells
      per dimension.
    batch_size: An `int`.

  Returns:
    An `int` tensor of shape [num_cells[0], num_cells[1], 2].
  """
  with tf.compat.v1.name_scope(
      name, "build grid ds", [sorted_keys, num_cells, batch_size]):
    return tfg_custom_ops.build_grid_ds(
      sorted_keys,
      num_cells,
      num_cells,
      batch_size)
tf.no_gradient('BuildGridDs')


def find_neighbors(grid,
                   point_cloud_centers,
                   radii,
                   max_neighbors=0,
                   name=None):
  """ Method to find the neighbors of a center point cloud in another
  point cloud.

  Args:
    grid: A Grid instance, from which the neighbors are chosen.
    point_cloud_centers: A `PointCloud` instance, containing the center points.
    radii: An `float` tensor of shape [D], the radii to select neighbors from.
    max_neighbors: An `int`, if `0` all neighbors are selected.

  Returns:
  center_neigh_ranges: An `int` tensor of shape [N], end of the ranges per
      center point. You can get the neighbor ids of point `i` (i>0) with
        `neighbors[center_neigh_ranges[i-1]:center_neigh_ranges[i]]`.
  neighbors: An `int` tensor of shape [M, 2], indices of the neighbor point and
      the center for each neighbor. Follows the order of `grid._sorted_points`.
  """
  with tf.compat.v1.name_scope(
      name, "find neighbours",
      [grid, point_cloud_centers, radii, max_neighbors]):
    return tfg_custom_ops.find_neighbors(
      point_cloud_centers._points,
      point_cloud_centers._batch_ids,
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
  """ Method to sample the points of a point cloud.

  Args:
    neighborhood: A `Neighborhood` instance, which contains a point cloud with
      its neighbors.
    sample_mode: A `SampleMode` value.

  Returns:
    sampled_points: A `float` tensor of shape [S, D], the sampled points.
    sampled_batch_ids: An `int` tensor of shape [S], the batch ids.
    sampled_indices: An `int` tensor of shape [S], the indices to the
      unsampled points.
      Following the order of neighborhood._grid._sorted_points.
  """
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
  """ Method to compute the density distribution inside the neighborhoods of a
  point cloud in euclidean space using kernel density estimation (KDE).

  Args:
    neighborhood: A `Neighborhood` instance.
    bandwidth: An `int` tensor of shape [D], the bandwidth of the KDE.
    mode: A `KDEMode` value.

  Returns:
    A `float` tensor of shape [S], the estimated density per center point.

  """
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
  """ Method to compute the convolution using a basis projection, i.e. a one
  hidden layer MLP, as convolution kernel and Monte-Carlo integration.

  Args:
    kernel_inputs: A float tensor of shape [N, L], the input to the kernel MLP.
    neighborhood: A `Neighborhood` instance.
    features: A `float`tensor of shape [N, C], the input features.
    basis: A list of two `tf.Variables`, the weights and biases of the
      hidden layer of the MLP.
    basis_type: An `int`, type of the activation function used.
      (RELU - 2, LRELU - 3, ELU - 4)

  Returns:
    A `float` tensor.
  """
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
