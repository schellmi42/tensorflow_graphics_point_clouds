/////////////////////////////////////////////////////////////////////////////
/// \file pooling_avg.cuh
///
/// \brief Declaraion of the CUDA operations to pool a set of points from
///     a point cloud. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
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