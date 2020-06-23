/////////////////////////////////////////////////////////////////////////////
/// Copyright 2020 Google LLC
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
///    https://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
/////////////////////////////////////////////////////////////////////////////
/// \brief Declaraion of the CUDA operations to compute the gradients of  
///     a basis projection operation.
/////////////////////////////////////////////////////////////////////////////

#ifndef BASIS_PROJ_GRADS_CUH_
#define BASIS_PROJ_GRADS_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute the gradients of a basis projection operation.
     *  @param  pBasisType                  Type of basis functions used.
     *  @param  pNumPts                     Number of points.
     *  @param  pNumSamples                 Number of samples.
     *  @param  pNumNeighbors               Number of neighbors.
     *  @param  pNumInFeatures              Number of input features.
     *  @param  pInKernelInGPUPtr           Input gpu pointer to the array
     *      with the inputs to the kernel.
     *  @param  pInPtFeaturesGPUPtr         Input gpu pointer to the array 
     *      with the point features.
     *  @param  pInNeighborsGPUPtr          Input gpu pointer with the list
     *      of neighbors.
     *  @param  pInSampleNeighIGPUPtr       Input gpu pointer with the 
     *      last neighboring point for each sample.
     *  @param  pInBasisGPUPtr              Input gpu pointer with the basis 
     *      functions.
     *  @param  pInPDFsGPUPtr               Input gpu pointer with the
     *      pdf values for each neighbor.
     *  @param  pInGradGPUPtr               Input gpu pointer with the gradients.
     *  @param  pOutFeatGradsGPUPtr         Output gpu pointer in which
     *      the input feature gradients will be stored.
     *  @param  pOutBasisGradsGPUPtr        Output gpu pointer in which 
     *      the gradients of the basis functions will be stored.
     *  @param  pOutKernelsInGradsGPUPtr       Output gpu pointer in which
     *      the gradients of the kernel inputs will be stored.
     *  @param  pOutPDFGradsGPUPtr          Output gpu pointer in which
     *      the gradietns of the pdf will be stored.
     *  @paramt D                       Number of dimensions.
     *  @paramt K                       Number of basis functions.
     */
    template<int D, int K>
    void basis_proj_grads_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const BasisFunctType pBasisType,
        const unsigned int pNumPts,
        const unsigned int pNumSamples,
        const unsigned int pNumNeighbors,
        const unsigned int pNumInFeatures, 
        const float* pInKernelInGPUPtr,
        const float* pInPtFeaturesGPUPtr,
        const int* pInNeighborsGPUPtr,
        const int* pInSampleNeighIGPUPtr,
        const float* pInBasisGPUPtr,
        const float* pInPDFsGPUPtr,
        const float* pInGradGPUPtr,
        float* pOutFeatGradsGPUPtr,
        float* pOutBasisGradsGPUPtr,
        float* pOutKernelsInGradsGPUPtr,
        float* pOutPDFGradsGPUPtr);
}

#endif