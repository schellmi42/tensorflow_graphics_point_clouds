/////////////////////////////////////////////////////////////////////////////
/// \file count_neighbors.cuh
///
/// \brief Declaraion of the CUDA operations to count the neighbors for each
///         point.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef COUNT_NEIGHBORS_CUH_
#define COUNT_NEIGHBORS_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to count the number of neighbors.
     *  @param  pDevice             GPU device.
     *  @param  pNumSamples         Number of samples.
     *  @param  pNumRanges          Number of ranges per point.
     *  @param  pInGPUPtrSamples    Input pointer to the vector of samples 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrPts        Input pointer to the vector of point 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrRanges     Input pointer to the vector of sample 
     *      ranges on the GPU.
     *  @param  pInvRadii           Inverse of the radius used on the 
     *      search of neighbors in each dimension.
     *  @param  pOutGPUPtrNumNeighs Output pointer to the vector with the 
     *      number of neighbors for each sample on the GPU. The memory
     *      should be initialized to 0 outside this function.
     *  @tparam D                   Number of dimensions.
     */
    template<int D>
    void count_neighbors(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumSamples,
        const unsigned int pNumRanges,
        const float* pInGPUPtrSamples,
        const float* pInGPUPtrPts,
        const int* pInGPUPtrRanges,
        const float* pInGPUPtrInvRadii,
        int* pOutGPUPtrNumNeighs);
        
}

#endif