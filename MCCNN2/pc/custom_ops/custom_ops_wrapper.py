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


def compute_keys(pPointCloud, pAABB, pNumCells, pCellSize, name=None):
  with tf.compat.v1.name_scope(
      name, "compute keys", [pPointCloud, pAABB, pNumCells, pCellSize]):
    return tfg_custom_ops.compute_keys(
      pPointCloud.pts_,
      pPointCloud.batchIds_,
      pAABB.aabbMin_ / pCellSize,
      pNumCells,
      tf.math.reciprocal(pCellSize))
tf.no_gradient('ComputeKeys')


def build_grid_ds(pKeys, pNumCells, pbatch_size, name=None):
  with tf.compat.v1.name_scope(
      name, "build grid ds", [pKeys, pNumCells, pbatch_size]):
    return tfg_custom_ops.build_grid_ds(
      pKeys,
      pNumCells,
      pNumCells,
      pbatch_size)
tf.no_gradient('BuildGridDs')


def find_neighbors(pGrid, pPCSamples, pRadii, pMaxNeighbors, name=None):
  with tf.compat.v1.name_scope(
      name, "find neighbours", [pGrid, pPCSamples, pRadii, pMaxNeighbors]):
    return tfg_custom_ops.find_neighbors(
      pPCSamples.pts_,
      pPCSamples.batchIds_,
      pGrid.sortedPts_,
      pGrid.sortedKeys_,
      pGrid.fastDS_,
      pGrid.numCells_,
      pGrid.aabb_.aabbMin_ / pGrid.cellSizes_,
      tf.math.reciprocal(pGrid.cellSizes_),
      tf.math.reciprocal(pRadii),
      pMaxNeighbors)
tf.no_gradient('FindNeighbors')


def sampling(pNeighborhood, pSampleMode, name=None):
  with tf.compat.v1.name_scope(name, "sampling", [pNeighborhood, pSampleMode]):
    return tfg_custom_ops.sampling(
      pNeighborhood.grid_.sortedPts_,
      pNeighborhood.grid_.sortedBatchIds_,
      pNeighborhood.grid_.sortedKeys_,
      pNeighborhood.grid_.numCells_,
      pNeighborhood.neighbors_,
      pNeighborhood.samplesNeighRanges_,
      pSampleMode)
tf.no_gradient('sampling')


def compute_pdf(pNeighborhood, pBandwidth, pMode, name=None):
  with tf.compat.v1.name_scope(
      name, "compute pdf with point gradients",
      [pNeighborhood, pBandwidth, pMode]):
    return tfg_custom_ops.compute_pdf_with_pt_grads(
      pNeighborhood.grid_.sortedPts_,
      pNeighborhood.neighbors_,
      pNeighborhood.samplesNeighRanges_,
      tf.math.reciprocal(pBandwidth),
      tf.math.reciprocal(pNeighborhood.radii_),
      pMode)


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


def basis_proj(pKernelInputs, pNeighborhood, pInFeatures,
               pBasis, pBasisType):
  curPDF = pNeighborhood.pdf_
  return tfg_custom_ops.basis_proj(
      pKernelInputs,
      pInFeatures,
      pNeighborhood.originalNeighIds_,
      pNeighborhood.samplesNeighRanges_,
      curPDF,
      pBasis,
      pBasisType)


@tf.RegisterGradient("BasisProj")
def _basis_proj_grad(op, *grads):
  featGrads, basisGrads, kernelInGrads, pdfGrads = \
      tfg_custom_ops.basis_proj_grads(
          op.inputs[0], op.inputs[1], op.inputs[2],
          op.inputs[3], op.inputs[4], op.inputs[5],
          grads[0], op.get_attr("basis_type"))
  return [kernelInGrads, featGrads, None, None,
          pdfGrads, basisGrads]
