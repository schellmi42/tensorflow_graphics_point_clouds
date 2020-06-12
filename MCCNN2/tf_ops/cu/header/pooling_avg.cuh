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
/// \brief Declaraion of the CUDA operations to pool a set of points from
///     a point cloud. 
/////////////////////////////////////////////////////////////////////////////

#ifndef POOLING_AVG_CUH_
#define POOLING_AVG_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to pool a set of points from a point cloud.
     *  @param  pDevice                 Device.
     *  @param  pNumPts                 Number of input points.
     *  @param  pNumPooledPts           Number of pooled points.
     *  @param  pInKeysGPUPtr           Input pointer to the vector of keys  
     *      sorted from bigger to smaller residing on GPU memory.
     *  @param  pPts GPUPtr             Input point coordinates.
     *  @param  pNumCellsGPUPtr         Number of cells.
     *  @param  pUniqueKeyIndexs        Unique key indices.
     *  @param  pOutPtsGPUPtr           Output array with the new point
     *      coordinates.
     *  @param  pBatchIdsGPUPtr         Output array with the batch ids
     *      of the new points.
     *  @paramt D                       Number of dimensions.
     */
     template <int D>
    void pooling_avg_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumPts,
        const unsigned int pNumPooledPts,
        const mccnn::int64_m* pInKeysGPUPtr,
        const float* pPtsGPUPtr,
        const int* pNumCellsGPUPtr,
        const int* pUniqueKeyIndexs,
        float* pOutPtsGPUPtr,
        int* pBatchIdsGPUPtr);
}

#endif