/////////////////////////////////////////////////////////////////////////////
/// \file knn.cuh
///
/// \brief Declaraion of the CUDA operations to compute the knn from a 
///     regular grid. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef KNN_CUH_
#define KNN_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute the knn on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pKnn                    Number of knn.
     *  @param  pNumRanges              Number of ranges.
     *  @param  pNumSamples             Number of samples.
     *  @param  pInGPUPtrPts            Input array with the points.
     *  @param  pInGPUPtrSamples        Input array with the samples.
     *  @param  pInGPUPtrRanges         Input array with the search ranges.
     *  @param  pInGPUPtrInvRadii       Input array with the inverse of
     *      the radii.
     *  @param  pOutGPUPtrKnn           Output array with the knn.
     *  @paramt D                       Number of dimensions.             
     */
    template<int D>
    void compute_knn_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const int pKnn,
        const unsigned int pNumRanges,
        const unsigned int pNumSamples,
        const float* pInGPUPtrPts,
        const float* pInGPUPtrSamples,
        const float* pInGPUPtrInvRadii,
        const int* pInGPUPtrRanges,
        int* pOutGPUPtrKnn);
}

#endif