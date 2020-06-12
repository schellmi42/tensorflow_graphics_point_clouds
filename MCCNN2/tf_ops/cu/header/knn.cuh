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
/// \brief Declaraion of the CUDA operations to compute the knn from a 
///     regular grid. 
/////////////////////////////////////////////////////////////////////////////

#ifndef KNN_CUH_
#define KNN_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute the knn on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pKnn                    Number of knn.
     *  @param  pNumRanges              Number of ranges.
     *  @param  pNumSamples             Number of samples.
     *  @param  pInGPUPtrPts            Input array with the points.
     *  @param  pInGPUPtrSamples        Input array with the samples.
     *  @param  pInGPUPtrRanges         Input array with the search ranges.
     *  @param  pInGPUPtrInvRadii       Input array with the inverse of
     *      the radii.
     *  @param  pOutGPUPtrKnn           Output array with the knn.
     *  @paramt D                       Number of dimensions.             
     */
    template<int D>
    void compute_knn_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const int pKnn,
        const unsigned int pNumRanges,
        const unsigned int pNumSamples,
        const float* pInGPUPtrPts,
        const float* pInGPUPtrSamples,
        const float* pInGPUPtrInvRadii,
        const int* pInGPUPtrRanges,
        int* pOutGPUPtrKnn);
}

#endif