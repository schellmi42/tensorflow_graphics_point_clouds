/////////////////////////////////////////////////////////////////////////////
/// \file count_unique_keys.cuh
///
/// \brief Declaraion of the CUDA operations to count the unique keys  
///     in a sorted array of keys. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef COUNT_UNIQUE_KEYS_CUH_
#define COUNT_UNIQUE_KEYS_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to count the number of unique keys on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNumPts                 Number of points.
     *  @param  pInKeysGPUPtr           Input pointer to the vector of keys  
     *      sorted from bigger to smaller residing on GPU memory.
     *  @returns    Number of unique keys in the array.
     */
    unsigned int count_unique_keys_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumPts,
        const mccnn::int64_m* pInKeysGPUPtr);
}

#endif