/////////////////////////////////////////////////////////////////////////////
/// \file compute_keys.cuh
///
/// \brief Declaraion of the CUDA operations to compute the keys indices 
///     of a point cloud into a regular grid. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef COMPUTE_KEYS_CUH_
#define COMPUTE_KEYS_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute the keys on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNumPts                 Number of points.
     *  @param  pInGPUPtrPts            Input pointer to the vector of points 
     *      on the GPU.
     *  @param  pInGPUPtrBatchIds       Input pointer to the vector of batch 
     *      ids on the GPU.
     *  @param  pInGPUPtrSAABBMin       Input pointer to the vector of minimum 
     *      points of the bounding boxes on the GPU scaled by the inverse
     *      cell size.
     *  @param  pInGPUPtrNumCells       Input pointer to the vector of number  
     *      of cells on the GPU.
     *  @param  pInGPUPtrInvCellSizes      Input pointer to the vector with the 
     *      inverse sizes of each cell.
     *  @param  pInGPUpOutGPUPtrKeys    Output pointer to the vector of keys  
     *      on the GPU.
     *  @paramT D                       Number of dimensions.
     */
    template<int D>
    void compute_keys_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumPts,
        const float* pInGPUPtrPts,
        const int* pInGPUPtrBatchIds,
        const float* pInGPUPtrSAABBMin,
        const int* pInGPUPtrNumCells,
        const float* pInGPUPtrInvCellSizes,
        mccnn::int64_m* pOutGPUPtrKeys);
}

#endif