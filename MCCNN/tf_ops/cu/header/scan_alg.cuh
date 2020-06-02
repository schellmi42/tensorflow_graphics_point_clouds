/////////////////////////////////////////////////////////////////////////////
/// \file scan_alg.cuh
///
/// \brief Declaraion of the CUDA operations to execute the parallel scan
///         algorithm in an int array.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef SCAN_ALG_CUH_
#define SCAN_ALG_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to execute the parallel scan algorithm in an int array.
     *  @param  pDevice                 GPU device.
     *  @param  pNumElems               Number of elements in the array.
     *      The number of elements should be multiple of T*2.
     *  @param  pInGPUPtrElems          Input pointer to the array on 
     *      the GPU.
     *  @return The total accumulation of elements at the end of the array.
     */
    unsigned int scan_alg(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumElems,
        int* pInGPUPtrElems);
        
}

#endif