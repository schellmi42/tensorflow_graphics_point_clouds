/////////////////////////////////////////////////////////////////////////////
/// \file compute_pdf.cuh
///
/// \brief Declaraion of the CUDA operations to compute the pdf of each 
///     neighboring point. 
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
     *  Method to compute the pdfs on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pMode                   Mode used to compute the sigma of the KDE:
     *      - 0: Constant.
     *      - 1: Based on the number of points in the receptive field. 1/sqrt(N)
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumNeighbors           Number of neighbors.
     *  @param  pInGPUPtrInvRadii       Inverse radii used to select the neighbors.
     *  @param  pInGPUPtrInvBandwidth   Inverse bandwidth value used in mode 0 for 
     *      the KDE.
     *  @param  pInGPUPtrPts            Input pointer to the vector of point 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrNeighbors      Input pointer to the vector of neighbors
     *      on the GPU.
     *  @param  pInGPUPtrSampleNeighI   Input pointer to the vector of number of
     *      neighbors for each sample on the GPU.
     *  @param  pOutGPUPtrPDFs          Output pointer to the vector of pdfs  
     *      on the GPU.      
     *  @paramt D                       Number of dimensions.             
     */
    template<int D>
    void compute_pdf_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pMode,
        const unsigned int pNumSamples,
        const unsigned int pNumNeighbors,
        const float* pInGPUPtrInvRadii,
        const float* pInGPUPtrInvBandwidth,
        const float* pInGPUPtrPts,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrSampleNeighI,
        float* pOutGPUPtrPDFs);

    /**
     *  Method to compute the gradients of the points wrt the pdfs values.
     *  @param  pDevice                 Device.
     *  @param  pMode                   Mode used to compute the sigma of the KDE:
     *      - 0: Constant.
     *      - 1: Based on the number of points in the receptive field. 1/sqrt(N)
     *  @param  pNumPts                 Number of points.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumNeighbors           Number of neighbors.
     *  @param  pInGPUPtrInvRadii       Inverse radii used to select the neighbors.
     *  @param  pInGPUPtrInvBandwidth   Inverse bandwidth value used in mode 0 for 
     *      the KDE.
     *  @param  pInGPUPtrPts            Input pointer to the vector of point 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrNeighbors      Input pointer to the vector of neighbors
     *      on the GPU.
     *  @param  pInGPUPtrSampleNeighI   Input pointer to the vector of number of
     *      neighbors for each sample on the GPU.
     *  @param  pInGPUPtrPDFGrad        Input gradient for each pdf value.
     *  @param  pOutGPUPtrPtGrads       Output pointer to the vector of pdfs  
     *      on the GPU.      
     *  @paramt D                       Number of dimensions.             
     */
    template<int D>
    void compute_pdf_grads_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pMode,
        const unsigned int pNumPts,
        const unsigned int pNumSamples,
        const unsigned int pNumNeighbors,
        const float* pInGPUPtrInvRadii,
        const float* pInGPUPtrInvBandwidth,
        const float* pInGPUPtrPts,
        const int* pInGPUPtrNeighbors,
        const int* pInGPUPtrSampleNeighI,
        const float* pInGPUPtrPDFGrad,
        float* pOutGPUPtrPtGrads);
}

#endif