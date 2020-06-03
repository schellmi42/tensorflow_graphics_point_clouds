/////////////////////////////////////////////////////////////////////////////
/// \file store_pooled_pts.cuh
///
/// \brief Declaraion of the CUDA operations to store in memory the pooled
///     points.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef STORED_POOLED_PTS_CUH_
#define STORED_POOLED_PTS_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to pool a set of points from a point cloud.
     *  @param  pDevice                 Device.
     *  @param  pNumPts                 Number of input points.
     *  @param  pNumPooledPts           Number of pooled points.
     *  @param  pPtsGPUPtr              Input pointer to the gpu array with
     *      the point coordinates.
     *  @param  pBatchIdsGPUPtr         Input pointer to the gpu array with
     *      the batch ids.
     *  @param  pSelectedGPUPtr         Input pointer to the gpu array with
     *      the selected points.
     *  @param  pOutPtsGPUPtr           Output pointer to the gpu array with
     *      the selected point coordinates.
     *  @param  pOutBatchIdsGPUPtr      Output pointer to the gpu array with
     *      the selected batch ids.
     *  @param  pOutIndicesGPUPtr       Output pointer to the gpu array with
     *      the selected indices.
     *  @paramt D                       Number of dimensions.
     */
    template<int D>
    void store_pooled_pts_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumPts,
        const unsigned int pNumPooledPts,
        const float* pPtsGPUPtr,
        const int* pBatchIdsGPUPtr,
        const int* pSelectedGPUPtr,
        float* pOutPtsGPUPtr,
        int* pOutBatchIdsGPUPtr,
        int* pOutIndicesGPUPtr);
}

#endif