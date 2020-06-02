/////////////////////////////////////////////////////////////////////////////
/// \file cuda_kernel_utils.cuh
///
/// \brief Utilities for the cuda implementations of the tensor operations.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_KERNEL_UTILS_H_
#define CUDA_KERNEL_UTILS_H_

#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda_fp16.h"

namespace mccnn{

    ///////////////////////// DEVICE FUNCTIONS

    /**
     *  Function to compute the global index of the current thread.
     *  @return   Current thread index.
     */
    __device__ __forceinline__ unsigned long long int compute_global_index_gpu_funct()
    {
        return threadIdx.x + blockDim.x*blockIdx.x;
    }

    /**
     *  Function to compute the total number of threads in execution..
     *  @return   Total number of threads.
     */
    __device__ __forceinline__ unsigned long long int compute_total_threads_gpu_funct()
    {
        return gridDim.x*blockDim.x;
    }

    /**
     *  Function to do an atomic max operation on floats.
     *  @param  pAddress    Address in which we want to perform the atomic operation.
     *  @param  pVal        Value we want to input.
     *  @return Stored value.
     */
    __device__ static float atomicMax(float* pAddress, const float pVal)
    {
        int* address_as_i = (int*) pAddress;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fmaxf(pVal, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }
}

#endif