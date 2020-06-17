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
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "build")

sys.path.append(BASE_DIR)

MCCNN2_module = tf.load_op_library(os.path.join(BUILD_DIR, 'MCCNN2.so'))


def compute_keys(pPointCloud, pAABB, pNumCells, pCellSize, name=None):
  with tf.compat.v1.name_scope(
      name, "compute keys", [pPointCloud, pAABB, pNumCells, pCellSize]):
    return MCCNN2_module.compute_keys(
      pPointCloud.pts_,
      pPointCloud.batchIds_,
      pAABB.aabbMin_ / pCellSize,
      pNumCells,
      tf.math.reciprocal(pCellSize))
tf.no_gradient('ComputeKeys')


def build_grid_ds(pKeys, pNumCells, pbatch_size, name=None):
  with tf.compat.v1.name_scope(
      name, "build grid ds", [pKeys, pNumCells, pbatch_size]):
    return MCCNN2_module.build_grid_ds(
      pKeys,
      pNumCells,
      pNumCells,
      pbatch_size)
tf.no_gradient('BuildGridDs')


def find_neighbors(pGrid, pPCSamples, pRadii, pMaxNeighbors, name=None):
  with tf.compat.v1.name_scope(
      name, "find neighbours", [pGrid, pPCSamples, pRadii, pMaxNeighbors]):
    return MCCNN2_module.find_neighbors(
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


def pooling(pNeighborhood, pPoolMode, name=None):
  with tf.compat.v1.name_scope(name, "pooling", [pNeighborhood, pPoolMode]):
    return MCCNN2_module.pooling(
      pNeighborhood.grid_.sortedPts_,
      pNeighborhood.grid_.sortedBatchIds_,
      pNeighborhood.grid_.sortedKeys_,
      pNeighborhood.grid_.numCells_,
      pNeighborhood.neighbors_,
      pNeighborhood.samplesNeighRanges_,
      pPoolMode)
tf.no_gradient('Pooling')


# def knn(pGrid, pPCSamples, pRadii, pNumNeighs, pNearest):
#     curNeighs = pNumNeighs
#     if not(pNearest):
#         curNeighs *= -1
#     return MCCNN2_module.knn(
#         pPCSamples.pts_,
#         pPCSamples.batchIds_,
#         pGrid.sortedPts_,
#         pGrid.sortedKeys_,
#         pGrid.fastDS_,
#         pGrid.numCells_,
#         pGrid.aabb_.aabbMin_/pGrid.cellSizes_,
#         tf.math.reciprocal(pGrid.cellSizes_),
#         tf.math.reciprocal(pRadii),
#         curNeighs)
# tf.no_gradient('KnnGrid')

# def emd_approx(pDistances, pNeighIndexs, pBatchIds, pNumPts1, pNumPts2):
#     return tf.transpose(MCCNN2_module.emd_approx(
#             tf.transpose(pDistances),
#             tf.transpose(pNeighIndexs),
#             pBatchIds, pNumPts1, pNumPts2))
# tf.no_gradient('EmdApprox')

def compute_pdf(pNeighborhood, pBandwidth, pMode, name=None):
  with tf.compat.v1.name_scope(
      name, "compute pdf", [pNeighborhood, pBandwidth, pMode]):
    return MCCNN2_module.compute_pdf(
      pNeighborhood.grid_.sortedPts_,
      pNeighborhood.neighbors_,
      pNeighborhood.samplesNeighRanges_,
      tf.math.reciprocal(pBandwidth),
      tf.math.reciprocal(pNeighborhood.radii_),
      pMode)
tf.no_gradient('ComputePdf')


def compute_pdf_with_pt_grads(pNeighborhood, pBandwidth, pMode, name=None):
  with tf.compat.v1.name_scope(
      name, "compute pdf with point gradients",
      [pNeighborhood, pBandwidth, pMode]):
    return MCCNN2_module.compute_pdf_with_pt_grads(
      pNeighborhood.grid_.sortedPts_,
      pNeighborhood.neighbors_,
      pNeighborhood.samplesNeighRanges_,
      tf.math.reciprocal(pBandwidth),
      tf.math.reciprocal(pNeighborhood.radii_),
      pMode)


@tf.RegisterGradient("ComputePdfWithPtGrads")
def _compute_pdf_grad(op, *grads):
  inPtsGrad = MCCNN2_module.compute_pdf_pt_grads(
    op.inputs[0],
    op.inputs[1],
    op.inputs[2],
    op.inputs[3],
    op.inputs[4],
    grads[0],
    op.get_attr("mode"))
  return [inPtsGrad, None, None, None, None]


# def compute_topo_dist(pGraph,
#                       pNeighborhood,
#                       pMaxDistance,
#                       pConstEdge=False):
#     intConstEdge = 0
#     if pConstEdge:
#         intConstEdge = 1
#     return MCCNN2_module.compute_topo_dist(
#         pNeighborhood.pcSamples_.pts_,
#         pNeighborhood.originalNeighIds_,
#         pGraph.neighbors_,
#         pGraph.nodeStartIndexs_,
#         pMaxDistance,
#         intConstEdge)
# tf.no_gradient('ComputeTopoDist')


# def compute_smooth_weights(pNeighborhood, pRadius):
#     return MCCNN2_module.compute_smooth_w(
#         pNeighborhood.grid_.sortedPts_,
#         pNeighborhood.pcSamples_.pts_,
#         pNeighborhood.neighbors_,
#         pNeighborhood.samplesNeighRanges_,
#         tf.math.reciprocal(pRadius))
# tf.no_gradient('ComputeSmoothW')


# def compute_smooth_weights_with_pt_grads(pNeighborhood, pRadius):
#     return MCCNN2_module.compute_smooth_w_with_pt_grads(
#         pNeighborhood.grid_.sortedPts_,
#         pNeighborhood.pcSamples_.pts_,
#         pNeighborhood.neighbors_,
#         pNeighborhood.samplesNeighRanges_,
#         tf.math.reciprocal(pRadius))
# @tf.RegisterGradient("ComputeSmoothWWithPtGrads")
# def _compute_smooth_w_grad(op, *grads):
#     inPtsGrad, inSampleGrad = MCCNN2_module.compute_smooth_w_pt_grads(
#         op.inputs[0],
#         op.inputs[1],
#         op.inputs[2],
#         op.inputs[3],
#         op.inputs[4],
#         grads[0])
#     return [inPtsGrad, inSampleGrad, None, None, None]


# def compute_protein_pooling(pGraph):
#     return MCCNN2_module.protein_pooling(
#         pGraph.neighbors_,
#         pGraph.nodeStartIndexs_)
# tf.no_gradient('ProteinPooling')


# def compute_graph_aggregation(pGraph, pFeatures, pNormalize):
#     if pNormalize:
#         inNorm = 1
#     else:
#         inNorm = 0
#     return MCCNN2_module.graph_aggregation(
#         pFeatures, pGraph.neighbors_,
#         pGraph.nodeStartIndexs_, inNorm)
# @tf.RegisterGradient("GraphAggregation")
# def _compute_graph_aggregation_grad(op, *grads):
#     outGrads = MCCNN2_module.graph_aggregation_grads(
#         grads[0], op.inputs[1], op.inputs[2],
#         op.get_attr("normalize"))
#     return [outGrads, None, None]


# def collapse_edges(pEdgeSortedIds, pEdgeIds, pStartNodeIds):
#     return MCCNN2_module.collapse_edges(
#         pEdgeSortedIds, pEdgeIds, pStartNodeIds)
# tf.no_gradient('CollapseEdges')


def basis_proj(pNeighborhood, pInFeatures,
        pBasis, pBasisType):
    if pNeighborhood.smoothW_ is None:
        curPDF = pNeighborhood.pdf_
    else:
        curPDF = pNeighborhood.pdf_ * tf.math.reciprocal(
            pNeighborhood.smoothW_)
    return MCCNN2_module.basis_proj(
        pNeighborhood.grid_.pointCloud_.pts_,
        pInFeatures,
        pNeighborhood.pcSamples_.pts_,
        pNeighborhood.originalNeighIds_,
        pNeighborhood.samplesNeighRanges_,
        tf.math.reciprocal(pNeighborhood.radii_),
        curPDF,
        pBasis,
        pBasisType,
        True)
@tf.RegisterGradient("BasisProj")
def _basis_proj_grad(op, *grads):
    featGrads, basisGrads, pointGrads, sampleGrads, pdfGrads = \
        MCCNN2_module.basis_proj_grads_with_pt_grads(
        op.inputs[0], op.inputs[1], op.inputs[2],
        op.inputs[3], op.inputs[4], op.inputs[5],
        op.inputs[6], op.inputs[7],
        grads[0], op.get_attr("basis_type"))
    return [pointGrads, featGrads, sampleGrads, None, None,
        None, pdfGrads, basisGrads]


# def basis_proj_bilateral(pNeighborhood, pNeighVals, pInFeatures,
#         pBasis, pBasisType, pPtGrads):
#     if pNeighborhood.smoothW_ is None:
#         curPDF = pNeighborhood.pdf_
#     else:
#         curPDF = pNeighborhood.pdf_ * tf.math.reciprocal(
#             pNeighborhood.smoothW_)
#     return MCCNN2_module.basis_proj_bil(
#         pNeighborhood.grid_.sortedPts_,
#         pInFeatures,
#         pNeighborhood.pcSamples_.pts_,
#         pNeighborhood.neighbors_,
#         pNeighborhood.samplesNeighRanges_,
#         tf.math.reciprocal(pNeighborhood.radii_),
#         curPDF,
#         pNeighVals,
#         pBasis,
#         pBasisType,
#         pPtGrads)
# @tf.RegisterGradient("BasisProjBil")
# def _basis_proj_bilateral_grad(op, *grads):
#     if op.get_attr("pt_grads"):
#         featGrads, basisGrads, pointGrads, sampleGrads, pdfGrads, \
#             neighGrads = \
#             MCCNN2_module.basis_proj_bil_grads_with_pt_grads(
#             op.inputs[0], op.inputs[1], op.inputs[2],
#             op.inputs[3], op.inputs[4], op.inputs[5],
#             op.inputs[6], op.inputs[7], op.inputs[8],
#             grads[0], op.get_attr("basis_type"))
#     else:
#         pointGrads = None
#         sampleGrads = None
#         pdfGrads = None
#         neighGrads = None
#         featGrads, basisGrads = MCCNN2_module.basis_proj_bil_grads(
#             op.inputs[0], op.inputs[1], op.inputs[2],
#             op.inputs[3], op.inputs[4], op.inputs[5],
#             op.inputs[6], op.inputs[7], op.inputs[8],
#             grads[0], op.get_attr("basis_type"))
#     return [pointGrads, featGrads, sampleGrads, None, None,
#         None, pdfGrads, neighGrads, basisGrads]


# def ray_mesh_intersec(pRayOrigins, pRayDirs, pMeshVox):
#     return MCCNN2_module.ray_mesh_intersection(
#         pRayOrigins,
#         pRayDirs,
#         pMeshVox.vertexs_,
#         pMeshVox.faces_,
#         pMeshVox.startVoxelIndexs_,
#         pMeshVox.faceIndexes_,
#         pMeshVox.coordMin_,
#         pMeshVox.coordMax_,
#         pMeshVox.cellSize_)
# tf.no_gradient('RayMeshIntersection')
