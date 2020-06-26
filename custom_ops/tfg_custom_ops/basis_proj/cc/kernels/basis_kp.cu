/////////////////////////////////////////////////////////////////////////////
/// Copyright 2020 Google LLC
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
///    https://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
/////////////////////////////////////////////////////////////////////////////
/// \brief Basis kernel points.
/////////////////////////////////////////////////////////////////////////////

#include "defines.hpp"
#include "cuda_kernel_utils.cuh"
#include "math_helper.cuh"
#include "basis/basis_kp.cuh"
#include "basis/basis_utils.cuh"

/**
 *  Template to define the type of correlation used.
 */
template<int D, int C> 
struct correlation_type_struct{
    __forceinline__ __device__ float forward(
        const float pKernelExtend,
        const float pDist){
            return 0.0f;
        }

    __forceinline__ __device__ void backward(
        const float pKernelExtend,
        const float pDist,
        const float* __restrict__ pKPtDiff,
        const float pInvPDF,
        const float pInGradient,
        float* __restrict__ pOutGrads,
        float* __restrict__ pOutPtGrads){}
};

// C = 0 : LINEAR
template<int D> 
struct correlation_type_struct<D, 0>{
    __forceinline__ __device__ float forward(
        const float pKernelExtend,
        const float pDist){
            return  MCCNN_MAX(1.0 - pDist*pKernelExtend, 0.0);
        }

    __forceinline__ __device__ void backward(
        const float pKernelExtend,
        const float pDist,
        const float* __restrict__ pKPtDiff,
        const float pInvPDF,
        const float pInGradient,
        float* __restrict__ pOutGrads,
        float* __restrict__ pOutPtGrads){

            float weightVal = this->forward(pKernelExtend, pDist);
            //If there is a gradient.
            if(weightVal > 0.0f){
#pragma unroll
                for(int j = 0; j < D; ++j){
                    float grad = (pKPtDiff[j]*pKernelExtend*pInvPDF*pInGradient)/(pDist+1e-8);
                    pOutGrads[j*blockDim.x] -= grad;
                    pOutPtGrads[j*blockDim.x] = grad;
                }
                pOutGrads[D*blockDim.x] -= pDist*pInvPDF*pInGradient;
                pOutPtGrads[D*blockDim.x] = -weightVal*pInvPDF*pInvPDF*pInGradient;
            }
        }
};

// C = 1 : EXPONENTIAL
template<int D> 
struct correlation_type_struct<D, 1>{
    __forceinline__ __device__ float forward(
        const float pKernelExtend,
        const float pDist){
            return expf(-(pDist*pDist*pKernelExtend));
        }

    __forceinline__ __device__ void backward(
        const float pKernelExtend,
        const float pDist,
        const float* __restrict__ pKPtDiff,
        const float pInvPDF,
        const float pInGradient,
        float* __restrict__ pOutGrads,
        float* __restrict__ pOutPtGrads){

            float expRes = this->forward(pKernelExtend, pDist);
#pragma unroll
            for(int j = 0; j < D; ++j){
                float grad = (expRes*pKernelExtend*2.0*pKPtDiff[j]*pInvPDF*pInGradient);
                pOutGrads[j*blockDim.x] -= grad;
                pOutPtGrads[j*blockDim.x] = grad;
            }
            pOutGrads[D*blockDim.x] += expRes*pInvPDF*(-pDist*pDist)*pInGradient;
            pOutPtGrads[D*blockDim.x] = -expRes*pInvPDF*pInvPDF*pInGradient;
        }
};

template<int D, int K, int C>
__global__ void compute_kp_basis_proj_pt_coords(
    const unsigned int pNumNeighbors,       
    const float* __restrict__ pInKernelInGPUPtr,
    const float* __restrict__ pInPDFsGPUPtr,
    const float* __restrict__ pInBasisGPUPtr,
    float* __restrict__ pOutProjGPUPtr)
{
    //Shared memory to store the kernel points.
    extern __shared__ float kernelPts[];

    //Create the struct to compute the kernel point correlation.
    correlation_type_struct<D, C> ptCorr;

    //Load the kernel point centers.
#pragma unroll(2)
    for(int i = threadIdx.x; i < K*(D+1); i+=blockDim.x)
        kernelPts[i] = pInBasisGPUPtr[i];

    __syncthreads();

    //Get usefull indices.
    const unsigned int initThreadIndex = mccnn::compute_global_index_gpu_funct();
    const unsigned int totalNumThreads = mccnn::compute_total_threads_gpu_funct(); 

    for(unsigned int curIter = initThreadIndex; 
        curIter < pNumNeighbors; curIter += totalNumThreads)
    {
        //Compute the pt difference.
        mccnn::fpoint<D> curKernelIns(&pInKernelInGPUPtr[curIter*D]);

        //Compute the pdf inverse.                
        float weightVal = 1.0f/(pInPDFsGPUPtr[curIter]);

        //Compute the projection of each basis.
        for(int i = 0; i < K; ++i){
            float sum = 0.0f;
#pragma unroll
            for(int j = 0; j < D; ++j){
                float auxDiff = kernelPts[i*(D+1) + j] - curKernelIns[j];
                sum += auxDiff*auxDiff;
            }

            float dist = sqrt(sum);
            float corr = ptCorr.forward(kernelPts[i*(D+1) + D], dist);
            pOutProjGPUPtr[curIter*K + i] = corr*weightVal;
        }
    }
}

template<int D, int K, int C>
__global__ void compute_kp_basis_proj_pt_coords_grads(
    const unsigned int pNumNeighbors,       
    const float* __restrict__ pInKernelInGPUPtr,
    const float* __restrict__ pInPDFsGPUPtr,
    const float* __restrict__ pInBasisGPUPtr,
    const float* __restrict__ pInGradsGPUPtr,
    float* __restrict__ pOutBasisGradsGPUPtr,
    float* __restrict__ pOutKernelInGradsGPUPtr,
    float* __restrict__ pOutPDFGradsGPUPtr)
{
    //Shared memory to store the kernel points.
    extern __shared__ float sharedMem[];

    //Create the struct to compute the kernel point correlation.
    correlation_type_struct<D, C> ptCorr;

    //Compute usefull indices.
    int totalExecThreads = pNumNeighbors*K;
    totalExecThreads += (totalExecThreads%blockDim.x != 0)?
        blockDim.x-totalExecThreads%blockDim.x:0;
    int groupId = threadIdx.x/K;
    int kpIndex = threadIdx.x%K;
    int groupsXBlock = blockDim.x/K;

    //Get the pointers to shared memory.
    float* kernelPts = sharedMem;
    float* accumGrads = &sharedMem[K*(D+1)];
    float* sharedKernelIns = &sharedMem[K*(D+1) + blockDim.x*(D+1)];
    float* accumPtGrads = &sharedMem[K*(D+1) + blockDim.x*(D+1) + groupsXBlock*D];

    //Load the kernel point centers.
#pragma unroll(2)
    for(int i = threadIdx.x; i < K*(D+1); i+=blockDim.x)
        kernelPts[i] = pInBasisGPUPtr[i];

#pragma unroll
    for(int i = 0; i < D+1; ++i)
        accumGrads[i*blockDim.x + threadIdx.x] = 0.0f;

    //Get usefull indices.
    const int initThreadIndex = mccnn::compute_global_index_gpu_funct();
    const int totalNumThreads = mccnn::compute_total_threads_gpu_funct(); 

    for(int curIter = initThreadIndex; 
        curIter < totalExecThreads; 
        curIter += totalNumThreads)
    {
        //Get indices to the point and sample.
        int neighIndex = curIter/K;
        float inGradient = 0.0f;

        if(neighIndex < pNumNeighbors){

            //Compute the pt difference.
            if(kpIndex < D){
                sharedKernelIns[groupId*D + kpIndex] = 
                    pInKernelInGPUPtr[neighIndex*D + kpIndex];
            }

            //Get the gradient.
            inGradient = pInGradsGPUPtr[neighIndex*K + kpIndex];
        }

        __syncthreads();

        if(neighIndex < pNumNeighbors){
            //Compute the pdf inverse.                
            float invPdf = 1.0f/(pInPDFsGPUPtr[neighIndex]);

            //Compute the projection of each basis.
            float sum = 0.0f;
            float kPtDiff[D];
#pragma unroll
            for(int j = 0; j < D; ++j){
                kPtDiff[j] = kernelPts[kpIndex*(D+1) + j] - sharedKernelIns[groupId*D + j];
                sum += kPtDiff[j]*kPtDiff[j];
            }

            float dist = sqrt(sum);
            //TODO - Add kahan summation, but requires more shared memory.
            ptCorr.backward(
                kernelPts[kpIndex*(D+1) + D],
                dist, kPtDiff,
                invPdf, inGradient,
                &accumGrads[threadIdx.x],
                &accumPtGrads[threadIdx.x]);
        }

        __syncthreads();

        if(neighIndex < pNumNeighbors && kpIndex < (D+1)){
            float accumVal = 0.0f;
#pragma unroll
            for(int j = 0; j < K; ++j){
                accumVal += accumPtGrads[groupId*K + kpIndex*blockDim.x + j];
            }
            if(kpIndex < D)
                pOutKernelInGradsGPUPtr[neighIndex*D+kpIndex] = accumVal;
            else
                pOutPDFGradsGPUPtr[neighIndex] = accumVal;
        }

        __syncthreads();
    }

    //Save the gradient into memory.
    for(int i = threadIdx.x; i < K*(D+1); i+=blockDim.x){
        int dimension = i/K;
        int kpoint = i%K;
        float accumVal = 0.0f;
#pragma unroll(2)
        for(int j = 0; j < groupsXBlock; ++j){
            accumVal += accumGrads[dimension*blockDim.x + j*K + kpoint];
        }
        atomicAdd(&pOutBasisGradsGPUPtr[kpoint*(D+1) + dimension], accumVal);
    }
}

/////////////////// CLASS DEFINITION

namespace mccnn{
        
    template<int D, int K>
    KPBasis<D, K>::KPBasis(KPBasis::PointCorrelation ptCorr)
        :BasisInterface<D, K>(), ptCorr_(ptCorr)
    {
    }

    template<int D, int K>
    KPBasis<D, K>::~KPBasis(void)
    {
    }

    template<int D, int K>
    void KPBasis<D, K>::compute_basis_proj_pt_coords(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNeighbors,       
        const float* pInKernelInGPUPtr,
        const float* pInPDFsGPUPtr,
        const float* pInBasisGPUPtr,
        float* pOutProjGPUPtr)
    {
        //Get the device properties.
        const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

        //Get information of the Device.
        unsigned int numMP = gpuProps.numMPs_;

        //Get the cuda stream.
        auto cudaStream = pDevice->getCUDAStream();

        //Define the block size.
        unsigned int blockSize = 64;

        //Get the current function pointer.
        const void* cFunct = nullptr;
        if(ptCorr_ == KPBasis<D, K>::PointCorrelation::LINEAR){
            cFunct = (const void*)compute_kp_basis_proj_pt_coords<D, K, 0>;
        }else if(ptCorr_ == KPBasis<D, K>::PointCorrelation::GAUSS){
            cFunct = (const void*)compute_kp_basis_proj_pt_coords<D, K, 1>;
        }

#ifdef DEBUG_INFO
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, cudaStream);
#endif
        //Calculate the shared memory needed.
        unsigned int sharedMemSize = (K*(D+1)*sizeof(float));

        //Compute the number of blocks
        unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
            blockSize, cFunct, sharedMemSize);
        pDevice->check_error(__FILE__, __LINE__);

        //Calculate the total number of blocks to execute.
        unsigned int execBlocks = pNumNeighbors/blockSize;
        execBlocks += (pNumNeighbors%blockSize != 0)?1:0;
        unsigned int totalNumBlocks = numMP*numBlocks;
        totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;
        
        //Execute the kernel extensions.
        if(ptCorr_ == KPBasis<D, K>::PointCorrelation::LINEAR){
            compute_kp_basis_proj_pt_coords<D, K, 0>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, pInKernelInGPUPtr,
                pInPDFsGPUPtr, pInBasisGPUPtr, pOutProjGPUPtr);
        }else if(ptCorr_ == KPBasis<D, K>::PointCorrelation::GAUSS){
            compute_kp_basis_proj_pt_coords<D, K, 1>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, pInKernelInGPUPtr,
                pInPDFsGPUPtr, pInBasisGPUPtr, pOutProjGPUPtr);
        }
        pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
        cudaEventRecord(stop, cudaStream);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        struct cudaFuncAttributes funcAttrib;
        cudaFuncGetAttributes(&funcAttrib, cFunct);
        float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

        fprintf(stderr, "### KP BASIS PROJ ###\n");
        fprintf(stderr, "Num basis: %d\n", K);
        fprintf(stderr, "Local memory: %d\n", (int)funcAttrib.localSizeBytes);
        fprintf(stderr, "Constant memory: %d\n", (int)funcAttrib.constSizeBytes);
        fprintf(stderr, "Num reg kernel: %d\n", funcAttrib.numRegs);
        fprintf(stderr, "Shared memory kernel: %d\n", sharedMemSize);
        fprintf(stderr, "Num neighbors: %d\n", pNumNeighbors);
        fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
        fprintf(stderr, "Execution time: %f\n", milliseconds);
        fprintf(stderr, "\n");
#endif
    }

    template<int D, int K>
    void KPBasis<D, K>::compute_grads_basis_proj_pt_coords(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNeighbors,       
        const float* pInKernelInGPUPtr,
        const float* pInPDFsGPUPtr,
        const float* pInBasisGPUPtr,
        const float* pInGradsGPUPtr,
        float* pOutBasisGradsGPUPtr,
        float* pOutKernelInsGradsGPUPtr,
        float* pOutPDFGradsGPUPtr)
    {
        //Get the device properties.
        const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

        //Get information of the Device.
        unsigned int numMP = gpuProps.numMPs_;

        //Get the cuda stream.
        auto cudaStream = pDevice->getCUDAStream();

        //Define the block size.
        unsigned int blockSize = 64;

        //Get the current function pointer.
        const void* cFunct = nullptr;
        if(ptCorr_ == KPBasis<D, K>::PointCorrelation::LINEAR){
            cFunct = (const void*)compute_kp_basis_proj_pt_coords_grads<D, K, 0>;
        }else if(ptCorr_ == KPBasis<D, K>::PointCorrelation::GAUSS){
            cFunct = (const void*)compute_kp_basis_proj_pt_coords_grads<D, K, 1>;
        }

#ifdef DEBUG_INFO
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, cudaStream);
#endif

        //Calculate the shared memory needed.
        unsigned int sharedMemSize = 
            (K*(D+1) + blockSize*(D+1)*2 + (blockSize/K)*D)*sizeof(float);

        //Compute the number of blocks
        unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
            blockSize, cFunct, sharedMemSize);
        pDevice->check_error(__FILE__, __LINE__);

        //Calculate the total number of blocks to execute.
        unsigned int execBlocks = (pNumNeighbors*K)/blockSize;
        execBlocks += ((pNumNeighbors*K)%blockSize != 0)?1:0;
        unsigned int totalNumBlocks = numMP*numBlocks;
        totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;
        
        //Execute the kernel extensions.
        if(ptCorr_ == KPBasis<D, K>::PointCorrelation::LINEAR){
            compute_kp_basis_proj_pt_coords_grads<D, K, 0>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInKernelInGPUPtr, pInPDFsGPUPtr, 
                    pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutKernelInsGradsGPUPtr, pOutPDFGradsGPUPtr);
        }else if(ptCorr_ == KPBasis<D, K>::PointCorrelation::GAUSS){
            compute_kp_basis_proj_pt_coords_grads<D, K, 1>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInKernelInGPUPtr, pInPDFsGPUPtr, 
                    pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutKernelInsGradsGPUPtr, pOutPDFGradsGPUPtr);
        }
        
        pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
        cudaEventRecord(stop, cudaStream);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        struct cudaFuncAttributes funcAttrib;
        cudaFuncGetAttributes(&funcAttrib, cFunct);
        float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

        fprintf(stderr, "### KP BASIS PROJ GRADS ###\n");
        fprintf(stderr, "Num basis: %d\n", K);
        fprintf(stderr, "Local memory: %d\n", (int)funcAttrib.localSizeBytes);
        fprintf(stderr, "Constant memory: %d\n", (int)funcAttrib.constSizeBytes);
        fprintf(stderr, "Num reg kernel: %d\n", funcAttrib.numRegs);
        fprintf(stderr, "Shared memory kernel: %d\n", sharedMemSize);
        fprintf(stderr, "Num neighbors: %d\n", pNumNeighbors);
        fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
        fprintf(stderr, "Execution time: %f\n", milliseconds);
        fprintf(stderr, "\n");
#endif
    }
}

//DECLARE THE VALID INSTANCES OF THE TEMPLATE CLASS
#define KP_BASIS_CLASS_DECL(D, K)    \
template class mccnn::KPBasis<D, K>;
DECLARE_TEMPLATE_DIMS_BASIS(KP_BASIS_CLASS_DECL)