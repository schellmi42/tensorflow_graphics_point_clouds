/////////////////////////////////////////////////////////////////////////////
/// \file store_neighbors.cuh
///
/// \brief Declaraion of the CUDA operations to store the neighbors for each
///         point.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef STORE_NEIGHBORS_CUH_
#define STORE_NEIGHBORS_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to store the number of neighbors.
     *  @param  pDevice                 GPU device.
     *  @param  pMaxNeighbors           Maximum number of neighbors. If zero or less, 
     *      there is not limit.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumRanges              Number of ranges per point.
     *  @param  pInGPUPtrSamples        Input pointer to the vector of samples 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrPts            Input pointer to the vector of point 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrRanges         Input pointer to the vector of sample 
     *      ranges on the GPU.
     *  @param  pInGPUPtrInvRadii       Inverse of the radius used on the 
     *      search of neighbors in each dimension.
     *  @param  pOutGPUPtrNumNeighsU    Input/Output pointer to the vector  
     *      with the number of neighbors for each sample without the limit of
     *      pMaxNeighbors.
     *  @param  pOutGPUPtrNumNeighs     Input/Output pointer to the vector  
     *      with the number of neighbors for each sample on the GPU.
     *  @param  pOutGPUPtrNeighs        Output pointer to the vector with the 
     *      number of neighbors for each sample on the GPU.
     *  @tparam D                       Number of dimensions.
     */
    template<int D>
    void store_neighbors(
        std::unique_ptr<IGPUDevice>& pDevice,
        const int pMaxNeighbors,
        const unsigned int pNumSamples,
        const unsigned int pNumRanges,
        const float* pInGPUPtrSamples,
        const float* pInGPUPtrPts,
        const int* pInGPUPtrRanges,
        const float* pInGPUPtrInvRadii,
        int* pOutGPUPtrNumNeighsU,
        int* pOutGPUPtrNumNeighs,
        int* pOutGPUPtrNeighs);
        
}

#endif