/////////////////////////////////////////////////////////////////////////////
/// \file pooling_pd.cuh
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

#ifndef COUNT_POOLING_PD_CUH_
#define COUNT_POOLING_PD_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to pool a set of points from a point cloud.
     *  @param  pDevice                 Device.
     *  @param  pNumPts                 Number of input points.
     *  @param  pNumUniqueKeys          Number of unique keys.
     *  @param  pUniqueKeyIndexs        Index of the start of each
     *      unique key.
     *  @param  pInKeysGPUPtr           Input pointer to the vector of keys  
     *      sorted from bigger to smaller residing on GPU memory.
     *  @param  pPtsGPUPtr              Input point coordinates.
     *  @param  pNeighbors              Input list of neighbors.
     *  @param  pNeighStartIndex        Input list of indicies of each list
     *      of neighbors.
     *  @param  pNumCellsGPUPtr         Number of cells.
     *  @param  pOutNumPooledPts        Output integer with the number
     *      of pooled points.
     *  @param  pOutPtsGPUPtr           Output array with the new point
     *      coordinates.
     *  @param  pBatchIdsGPUPtr         Output array with the batch ids
     *      of the new points.
     *  @paramt D                       Number of dimensions.
     */
    template <int D>
    void count_pooling_pd_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumPts,
        const unsigned int pNumUniqueKeys,
        const int* pUniqueKeyIndexs,
        const mccnn::int64_m* pInKeysGPUPtr,
        const float* pPtsGPUPtr,
        const int* pNeighbors,
        const int* pNeighStartIndex,
        const int* pNumCellsGPUPtr,
        int& pOutNumPooledPts,
        int* pSelectedGPUPtr);
}

#endif