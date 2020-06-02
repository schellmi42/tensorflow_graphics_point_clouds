/////////////////////////////////////////////////////////////////////////////
/// \file compute_smooth_weights.cuh
///
/// \brief Declaraion of the CUDA operations to compute the smooth weights
///      of each neighboring point. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef COMPUTE_PDF_CUH_
#define COMPUTE_PDF_CUH_

#include "gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute the smooth weights on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumNeighbors           Number of neighbors.
     *  @param  pInGPUPtrInvRadii       Inverse radii used to select the neighbors.
     *  @param  pInGPUPtrPts            Input pointer to the vector of point 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrSamples        Input pointer to the vector of point
     *      coordinates of the samples on the GPU.
     *  @param  pInGPUPtrNeighbors      Input pointer to the vector of neighbors
     *      on the GPU.
     *  @param  pInGPUPtrSampleNeighI   Input pointer to the vector of number of
     *      neighbors for each sample on the GPU.
     *  @param  pOutGPUPtrSmoothW       Output pointer to the vector of  
     *      smooth weights on the GPU.      
     *  @paramt D                       Number of dimensions.             
     */
    template<int D>
    void compute_smooth_weights_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumSamples,
        const unsigned int pNumNeighbors,
        const float* pInGPUPtrInvRadii,
        const float* pInGPUPtrPts,
        const float* pInGPUPtrSamples,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrSampleNeighI,
        float* pOutGPUPtrSmoothW);

    /**
     *  Method to compute the gradients of the points wrt the smooth weights.
     *  @param  pDevice                 Device.
     *  @param  pNumPts                 Number of points.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumNeighbors           Number of neighbors.
     *  @param  pInGPUPtrInvRadii       Inverse radii used to select the neighbors.
     *  @param  pInGPUPtrPts            Input pointer to the vector of point 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrSamples        Input pointer to the vector of point
     *      coordinates of the samples on the GPU.
     *  @param  pInGPUPtrNeighbors      Input pointer to the vector of neighbors
     *      on the GPU.
     *  @param  pInGPUPtrSampleNeighI   Input pointer to the vector of number of
     *      neighbors for each sample on the GPU.
     *  @param  pInGPUPtrSmoothWGrad    Input gradient for each smooth weight value.
     *  @param  pOutGPUPtrPtGrads       Output pointer to the vector of point  
     *      coordinate gradients.      
     *  @param  pOutGPUPtrSampleGrads   Output pointer to the vector of sample
     *      coordinate gradients.
     *  @paramt D                       Number of dimensions.             
     */
    template<int D>
    void compute_smooth_weights_grads_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumPts,
        const unsigned int pNumSamples,
        const unsigned int pNumNeighbors,
        const float* pInGPUPtrInvRadii,
        const float* pInGPUPtrPts,
        const float* pInGPUPtrSamples,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrSampleNeighI,
        const float* pInGPUPtrSmoothWGrad,
        float* pOutGPUPtrPtGrads,
        float* pOutGPUPtrSampleGrads);
}

#endif