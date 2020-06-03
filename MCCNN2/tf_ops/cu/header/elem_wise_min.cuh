/////////////////////////////////////////////////////////////////////////////
/// \file elem_wise_min.cuh
///
/// \brief Declaraion of the CUDA operations to perform a min operation 
///     element wise.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef ELEM_WISE_MIN_CUH_
#define ELEM_WISE_MIN_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to perform element wise minimum operation with a given
     *  minimum value.
     *  @param      pDevice                 Device.
     *  @param      pNumElements            Number of elements in the array.
     *  @param      pMinValue               Minimum value.
     *  @param      pValuesGPUPtr           Input/Output pointer to the vector of 
     *      values in GPU memory
     *  @rparamt    T                       Type of the elements.
     */
    template<class T>
    void elem_wise_min_value(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumElements,
        const T pMinValue,
        T* pValuesGPUPtr);
}

#endif