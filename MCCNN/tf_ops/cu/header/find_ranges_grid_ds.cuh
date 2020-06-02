/////////////////////////////////////////////////////////////////////////////
/// \file find_ranges_grid_ds.cuh
///
/// \brief Declaraion of the CUDA operations to find the ranges in the list
///     of points for a grid cell and its 26 neighbors.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef BUILD_GRID_DS_CUH_
#define BUILD_GRID_DS_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to find the ranges in the list of points for a grid cell 
     *  and its 26 neighbors.
     *  @param  pDevice                 Device.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumPts                 Number of points.
     *  @param  pLastDOffsets           Number of displacement in the last
     *      dimension in the positive and negative axis.
     *  @param  pNumOffsets             Number of offsets applied to  the 
     *      keys.
     *  @param  pInGPUPtrOffsets        List of offsets to apply.
     *  @param  pInGPUPtrSampleKeys     Input pointer to the vector of keys 
     *      of each sample on the GPU.
     *  @param  pInGPUPtrPtsKeys        Input pointer to the vector of keys 
     *      of each point on the GPU.
     *  @param  pInGPUPtrGridDS         Input grid acceleration data 
     *      structure.
     *  @param  pInGPUPtrNumCells       Input pointer to the vector of number  
     *      of cells on the GPU.
     *  @param  pOutGPUPtrRanges        Output pointer to the array containing
     *      the search ranges for each sample. 
     *  @paramT D                       Number of dimensions.
     */
    template<int D>
    void find_ranges_grid_ds_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumSamples, 
        const unsigned int pNumPts,
        const unsigned int pLastDOffsets,
        const unsigned int pNumOffsets,
        const int* pInGPUPtrOffsets,
        const mccnn::int64_m* pInGPUPtrSampleKeys,
        const mccnn::int64_m* pInGPUPtrPtsKeys,
        const int* pInGPUPtrGridDS,
        const int* pInGPUPtrNumCells,
        int* pOutGPUPtrRanges);

    /**
     *  Method to compute the total number of offsets
     *  to apply for each range search.
     *  @param  pNumDimensions  Number of dimensions.
     *  @param  pAxisOffset     Offset apply to each axis.
     *  @param  pOutVector      Output parameter with the 
     *      displacements applied to each axis.
     */
    unsigned int computeTotalNumOffsets(
        const unsigned int pNumDimensions,
        const unsigned int pAxisOffset,
        std::vector<int>& pOutVector);
}

#endif