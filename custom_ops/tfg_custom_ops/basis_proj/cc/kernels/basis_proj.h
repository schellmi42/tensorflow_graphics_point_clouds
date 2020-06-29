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
/// \brief Declaraion of the CUDA operations to project the input features
///     in a set of basis.
/////////////////////////////////////////////////////////////////////////////

#ifndef BASIS_PROJ_CUH_
#define BASIS_PROJ_CUH_

#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute a monte carlo convolution using an kernel sample.
     *  @param  pBasisType              Type of basis functions used.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumNeighbors           Number of neighbors.
     *  @param  pNumInFeatures          Number of input features.
     *  @param  pInKernelInGPUPtr       Input gpu pointer to the array
     *      with the inputs to the kernel.
     *  @param  pInPtFeaturesGPUPtr     Input gpu pointer to the array
     *      with the input features.
     *  @param  pInNeighborsGPUPtr      Input gpu pointer with the list
     *      of neighbors.
     *  @param  pInSampleNeighIGPUPtr   Input gpu pointer with the 
     *      last neighbor index for each sample.
     *  @param  pInBasisGPUPtr          Input gpu pointer with the basis
     *      functions.
     *  @param  pInPDFsGPUPtr           Input gpu pointer with the
     *      pdf values for each neighbor.
     *  @param  pOutFeaturesGPUPtr      Output gpu pointer in which
     *      the new features will be stored.
     *  @paramt D                       Number of dimensions.
     *  @paramt K                       Number of basis functions.
     */
    template<int D, int K>
    void basis_proj_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const BasisFunctType pBasisType,
        const unsigned int pNumSamples,
        const unsigned int pNumNeighbors,
        const unsigned int pNumInFeatures,
        const float* pInKernelInGPUPtr,
        const float* pInPtFeaturesGPUPtr,
        const int* pInNeighborsGPUPtr,
        const int* pInSampleNeighIGPUPtr,
        const float* pInBasisGPUPtr,
        const float* pInPDFsGPUPtr,
        float*  pOutFeaturesGPUPtr);
}

#endif