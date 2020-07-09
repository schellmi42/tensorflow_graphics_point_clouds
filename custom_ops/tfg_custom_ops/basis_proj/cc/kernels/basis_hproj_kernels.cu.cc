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
/// \brief Basis half projection.
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/nn_utils.h"

#include "tfg_custom_ops/basis_proj/cc/kernels/basis_hproj.h"
#include "tfg_custom_ops/basis_proj/cc/kernels/basis_utils.h"

template<int D, int K, int A>
__global__ void compute_hproj_basis_proj_pt_coords(
    const unsigned int pNumNeighbors,       
    const float* __restrict__ pInKernelInGPUPtr,
    const float* __restrict__ pInPDFsGPUPtr,
    const float* __restrict__ pInBasisGPUPtr,
    float* __restrict__ pOutProjGPUPtr)
{
    //Shared memory to store the kernel points.
    extern __shared__ float kernelPts[];

    //Create the struct to compute the activation function.
    mccnn::activation_function_struct<A> acFunc;

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
            for(int j = 0; j < D; ++j)
                sum += kernelPts[i*(D+1) + j]*curKernelIns[j];
            sum += kernelPts[i*(D+1) + D];
            pOutProjGPUPtr[curIter*K + i] = acFunc.forward(sum)*weightVal;
        }
    }
}


template<int D, int K, int A>
__global__ void compute_hproj_basis_proj_pt_coords_grads(
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

    //Create the struct to compute the activation function.
    mccnn::activation_function_struct<A> acFunc;

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
#pragma unroll
            for(int j = 0; j < D; ++j)
                sum += kernelPts[kpIndex*(D+1) + j]*sharedKernelIns[groupId*D + j];
            sum += kernelPts[kpIndex*(D+1) + D];
            float value = acFunc.forward(sum);

            //Compute the gradient before the projection.
            float curInGradient = inGradient * acFunc.backward(value) * invPdf;

            //Compute the gradients
            //TODO - Add kahan summation, but requires more shared memory.
#pragma unroll
            for(int j = 0; j < D; ++j){
                accumGrads[threadIdx.x + j*blockDim.x] += 
                    sharedKernelIns[groupId*D + j]*curInGradient;
                accumPtGrads[threadIdx.x + j*blockDim.x] = 
                    curInGradient*kernelPts[kpIndex*(D+1) + j];
            }
            accumGrads[threadIdx.x + D*blockDim.x] += curInGradient;//Bias
            accumPtGrads[threadIdx.x + D*blockDim.x] = -value*invPdf*invPdf*inGradient;//PDF
        }

        __syncthreads();

        if(neighIndex < pNumNeighbors && kpIndex < (D+1)){

            float accumVal = 0.0f;
#pragma unroll
            for(int j = 0; j < K; ++j){
                accumVal += accumPtGrads[groupId*K + kpIndex*blockDim.x + j];
            }
            if(kpIndex < D)
                pOutKernelInGradsGPUPtr[neighIndex*D + kpIndex] = accumVal;
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
    HProjBasis<D, K>::HProjBasis(HProjBasis::ActivationFunction pAcFunc)
        :BasisInterface<D, K>(), acFunc_(pAcFunc)
    {
    }

    template<int D, int K>
    HProjBasis<D, K>::~HProjBasis(void)
    {
    }

    template<int D, int K>
    void HProjBasis<D, K>::compute_basis_proj_pt_coords(
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
        if(acFunc_ == HProjBasis<D, K>::ActivationFunction::RELU){
            cFunct = (const void*)compute_hproj_basis_proj_pt_coords<D, K, 0>;
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::LRELU){
            cFunct = (const void*)compute_hproj_basis_proj_pt_coords<D, K, 1>;
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::ELU){
            cFunct = (const void*)compute_hproj_basis_proj_pt_coords<D, K, 2>;
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::EXP){
            cFunct = (const void*)compute_hproj_basis_proj_pt_coords<D, K, 3>;
        }

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
        if(acFunc_ == HProjBasis<D, K>::ActivationFunction::RELU){
            compute_hproj_basis_proj_pt_coords<D, K, 0>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, pInKernelInGPUPtr,
                pInPDFsGPUPtr, pInBasisGPUPtr, pOutProjGPUPtr);
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::LRELU){
            compute_hproj_basis_proj_pt_coords<D, K, 1>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, pInKernelInGPUPtr,
                pInPDFsGPUPtr, pInBasisGPUPtr, pOutProjGPUPtr);
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::ELU){
            compute_hproj_basis_proj_pt_coords<D, K, 2>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, pInKernelInGPUPtr,
                pInPDFsGPUPtr, pInBasisGPUPtr, pOutProjGPUPtr);
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::EXP){
            compute_hproj_basis_proj_pt_coords<D, K, 3>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, pInKernelInGPUPtr,
                pInPDFsGPUPtr, pInBasisGPUPtr, pOutProjGPUPtr);
        }
        pDevice->check_error(__FILE__, __LINE__);
    }

    template<int D, int K>
    void HProjBasis<D, K>::compute_grads_basis_proj_pt_coords(
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
        if(acFunc_ == HProjBasis<D, K>::ActivationFunction::RELU){
            cFunct = (const void*)compute_hproj_basis_proj_pt_coords_grads<D, K, 0>;
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::LRELU){
            cFunct = (const void*)compute_hproj_basis_proj_pt_coords_grads<D, K, 1>;
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::ELU){
            cFunct = (const void*)compute_hproj_basis_proj_pt_coords_grads<D, K, 2>;
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::EXP){
            cFunct = (const void*)compute_hproj_basis_proj_pt_coords_grads<D, K, 3>;
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
        if(acFunc_ == HProjBasis<D, K>::ActivationFunction::RELU){
            compute_hproj_basis_proj_pt_coords_grads<D, K, 0>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInKernelInGPUPtr, pInPDFsGPUPtr, 
                    pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutKernelInsGradsGPUPtr, pOutPDFGradsGPUPtr);
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::LRELU){
            compute_hproj_basis_proj_pt_coords_grads<D, K, 1>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInKernelInGPUPtr, pInPDFsGPUPtr, 
                    pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutKernelInsGradsGPUPtr, pOutPDFGradsGPUPtr);
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::ELU){
            compute_hproj_basis_proj_pt_coords_grads<D, K, 2>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInKernelInGPUPtr, pInPDFsGPUPtr, 
                    pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutKernelInsGradsGPUPtr, pOutPDFGradsGPUPtr);
        }else if(acFunc_ == HProjBasis<D, K>::ActivationFunction::EXP){
            compute_hproj_basis_proj_pt_coords_grads<D, K, 3>
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

        fprintf(stderr, "### HPROJ BASIS PROJ GRADS ###\n");
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
#define HPROJ_BASIS_CLASS_DECL(D, K)    \
template class mccnn::HProjBasis<D, K>;
DECLARE_TEMPLATE_DIMS_BASIS(HPROJ_BASIS_CLASS_DECL)