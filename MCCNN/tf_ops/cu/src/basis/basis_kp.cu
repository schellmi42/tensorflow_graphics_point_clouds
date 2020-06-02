/////////////////////////////////////////////////////////////////////////////
/// \file basis_kp.cuh
///
/// \brief Basis kernel points.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "defines.hpp"
#include "cuda_kernel_utils.cuh"
#include "math_helper.cuh"
#include "basis/basis_kp.cuh"
#include "basis/basis_utils.cuh"

/**
 *  Template to define the type of correlation used.
 */
template<int D, bool P, int C> 
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
        const float* __restrict__ pInvRadius,
        const float pInvPDF,
        const float pInGradient,
        float* __restrict__ pOutGrads,
        float* __restrict__ pOutPtGrads){}
};

// C = 0 : LINEAR
template<int D> 
struct correlation_type_struct<D, false, 0>{
    __forceinline__ __device__ float forward(
        const float pKernelExtend,
        const float pDist){
            return  MCCNN_MAX(1.0 - pDist*pKernelExtend, 0.0);
        }

    __forceinline__ __device__ void backward(
        const float pKernelExtend,
        const float pDist,
        const float* __restrict__ pKPtDiff,
        const float* __restrict__ pInvRadius,
        const float pInvPDF,
        const float pInGradient,
        float* __restrict__ pOutGrads,
        float* __restrict__ pOutPtGrads){

            float weightVal = this->forward(pKernelExtend, pDist);
            //If there is a gradient.
            if(weightVal > 0.0f){
#pragma unroll
                for(int j = 0; j < D; ++j){
                    pOutGrads[j*blockDim.x] -= 
                        (pKPtDiff[j]*pKernelExtend*pInvPDF*pInGradient)/(pDist+1e-8);
                }
                pOutGrads[D*blockDim.x] -= pDist*pInvPDF*pInGradient;
            }
        }
};


template<int D> 
struct correlation_type_struct<D, true, 0>{
    __forceinline__ __device__ float forward(
        const float pKernelExtend,
        const float pDist){
            return  MCCNN_MAX(1.0 - pDist*pKernelExtend, 0.0);
        }

    __forceinline__ __device__ void backward(
        const float pKernelExtend,
        const float pDist,
        const float* __restrict__ pKPtDiff,
        const float* __restrict__ pInvRadius,
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
                    pOutPtGrads[j*blockDim.x] = pInvRadius[j]*grad;
                    pOutPtGrads[(D+j)*blockDim.x] = -pInvRadius[j]*grad;
                }
                pOutGrads[D*blockDim.x] -= pDist*pInvPDF*pInGradient;
                pOutPtGrads[D*2*blockDim.x] = -weightVal*pInvPDF*pInvPDF*pInGradient;
            }
        }
};

// C = 1 : EXPONENTIAL
template<int D> 
struct correlation_type_struct<D, false, 1>{
    __forceinline__ __device__ float forward(
        const float pKernelExtend,
        const float pDist){
            return expf(-(pDist*pDist*pKernelExtend));
        }

    __forceinline__ __device__ void backward(
        const float pKernelExtend,
        const float pDist,
        const float* __restrict__ pKPtDiff,
        const float* __restrict__ pInvRadius,
        const float pInvPDF,
        const float pInGradient,
        float* __restrict__ pOutGrads,
        float* __restrict__ pOutPtGrads){

            float expRes = this->forward(pKernelExtend, pDist);
#pragma unroll
            for(int j = 0; j < D; ++j)
                pOutGrads[j*blockDim.x] -= 
                    (expRes*pKernelExtend*2.0*pKPtDiff[j]*pInvPDF*pInGradient);
            pOutGrads[D*blockDim.x] += expRes*pInvPDF*(-pDist*pDist)*pInGradient;
        }
};

template<int D> 
struct correlation_type_struct<D, true, 1>{
    __forceinline__ __device__ float forward(
        const float pKernelExtend,
        const float pDist){
            return expf(-(pDist*pDist*pKernelExtend));
        }

    __forceinline__ __device__ void backward(
        const float pKernelExtend,
        const float pDist,
        const float* __restrict__ pKPtDiff,
        const float* __restrict__ pInvRadius,
        const float pInvPDF,
        const float pInGradient,
        float* __restrict__ pOutGrads,
        float* __restrict__ pOutPtGrads){

            float expRes = this->forward(pKernelExtend, pDist);
#pragma unroll
            for(int j = 0; j < D; ++j){
                float grad = (expRes*pKernelExtend*2.0*pKPtDiff[j]*pInvPDF*pInGradient);
                pOutGrads[j*blockDim.x] -= grad;
                pOutPtGrads[j*blockDim.x] = pInvRadius[j]*grad;
                pOutPtGrads[(D+j)*blockDim.x] = -pInvRadius[j]*grad;
            }
            pOutGrads[D*blockDim.x] += expRes*pInvPDF*(-pDist*pDist)*pInGradient;
            pOutPtGrads[D*2*blockDim.x] = -expRes*pInvPDF*pInvPDF*pInGradient;
        }
};

/**
 *  Template to accumulate the point gradients.
 */
template<int D, int K, bool P> 
struct accum_pt_grads{

    __forceinline__ __device__ void accumulate(
        const int pOffset,
        const float* pSharedMem,
        float* __restrict__ pOutPtGrads,
        float* __restrict__ pOutSampleGrads,
        float* __restrict__ pOutPDFGrads){}
};

template<int D, int K> 
struct accum_pt_grads<D, K, true>{

    __forceinline__ __device__ void accumulate(
        const int pOffset,
        const float* __restrict__ pSharedMem,
        float* __restrict__ pOutPtGrads,
        float* __restrict__ pOutSampleGrads,
        float* __restrict__ pOutPDFGrads){
        float accumVal = 0.0f;
#pragma unroll
        for(int j = 0; j < K; ++j){
            accumVal += pSharedMem[pOffset*blockDim.x + j];
        }
        if(pOffset < D)
            atomicAdd(&pOutPtGrads[pOffset], accumVal);
        else if(pOffset < D*2)
            atomicAdd(&pOutSampleGrads[pOffset - D], accumVal);
        else
            pOutPDFGrads[0] = accumVal;
    }
};

template<int D, int K, int C>
__global__ void compute_kp_basis_proj_pt_coords(
    const unsigned int pNumNeighbors,       
    const mccnn::fpoint<D>* __restrict__ pInPtsGPUPtr,
    const mccnn::fpoint<D>* __restrict__ pInSamplesGPUPtr,
    const mccnn::fpoint<D>* __restrict__ pInInvRadiiGPUPtr,
    const int2* __restrict__ pInNeighborsGPUPtr,
    const float* __restrict__ pInPDFsGPUPtr,
    const float* __restrict__ pInBasisGPUPtr,
    float* __restrict__ pOutProjGPUPtr)
{
    //Shared memory to store the kernel points.
    extern __shared__ float kernelPts[];

    //Create the struct to compute the kernel point correlation.
    correlation_type_struct<D, false, C> ptCorr;

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
        //Get indices to the point and sample.
        int2 neighAndSampleIndices = pInNeighborsGPUPtr[curIter];

        //Compute the pt difference.
        mccnn::fpoint<D> ptDiff = (pInPtsGPUPtr[neighAndSampleIndices.x] - 
            pInSamplesGPUPtr[neighAndSampleIndices.y])*pInInvRadiiGPUPtr[0];

        //Compute the pdf inverse.                
        float weightVal = 1.0f/(pInPDFsGPUPtr[curIter]);

        //Compute the projection of each basis.
        for(int i = 0; i < K; ++i){
            float sum = 0.0f;
#pragma unroll
            for(int j = 0; j < D; ++j){
                float auxDiff = kernelPts[i*(D+1) + j] - ptDiff[j];
                sum += auxDiff*auxDiff;
            }

            float dist = sqrt(sum);
            float corr = ptCorr.forward(kernelPts[i*(D+1) + D], dist);
            pOutProjGPUPtr[curIter*K + i] = corr*weightVal;
        }
    }
}

template<int D, int K, int C, bool P>
__global__ void compute_kp_basis_proj_pt_coords_grads(
    const unsigned int pNumNeighbors,       
    const float* __restrict__ pInPtsGPUPtr,
    const float* __restrict__ pInSamplesGPUPtr,
    const float* __restrict__ pInInvRadiiGPUPtr,
    const int2* __restrict__ pInNeighborsGPUPtr,
    const float* __restrict__ pInPDFsGPUPtr,
    const float* __restrict__ pInBasisGPUPtr,
    const float* __restrict__ pInGradsGPUPtr,
    float* __restrict__ pOutBasisGradsGPUPtr,
    float* __restrict__ pOutPtsGradsGPUPtr,
    float* __restrict__ pOutSampleGradsGPUPtr,
    float* __restrict__ pOutPDFGradsGPUPtr)
{
    //Shared memory to store the kernel points.
    extern __shared__ float sharedMem[];

    //Create the struct to compute the kernel point correlation.
    correlation_type_struct<D, P, C> ptCorr;

    //Create the struct to compute point gradients.
    accum_pt_grads<D, K, P> ptGrads;

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
    float* sharedPtDiffs = &sharedMem[K*(D+1) + blockDim.x*(D+1)];
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
        int2 neighAndSampleIndices;
        int neighIndex = curIter/K;
        float inGradient = 0.0f;

        if(neighIndex < pNumNeighbors){
            neighAndSampleIndices = pInNeighborsGPUPtr[neighIndex];

            //Compute the pt difference.
            if(kpIndex < D){
                sharedPtDiffs[groupId*D + kpIndex] = 
                    (pInPtsGPUPtr[neighAndSampleIndices.x*D + kpIndex] -
                    pInSamplesGPUPtr[neighAndSampleIndices.y*D + kpIndex])*
                    pInInvRadiiGPUPtr[kpIndex];
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
                kPtDiff[j] = kernelPts[kpIndex*(D+1) + j] - sharedPtDiffs[groupId*D + j];
                sum += kPtDiff[j]*kPtDiff[j];
            }

            float dist = sqrt(sum);
            //TODO - Add kahan summation, but requires more shared memory.
            ptCorr.backward(
                kernelPts[kpIndex*(D+1) + D],
                dist, kPtDiff, pInInvRadiiGPUPtr,
                invPdf, inGradient,
                &accumGrads[threadIdx.x],
                &accumPtGrads[threadIdx.x]);
        }

        __syncthreads();

        if(neighIndex < pNumNeighbors && kpIndex < (D*2+1)){
            ptGrads.accumulate(kpIndex, &accumPtGrads[groupId*K],
                &pOutPtsGradsGPUPtr[neighAndSampleIndices.x*D],
                &pOutSampleGradsGPUPtr[neighAndSampleIndices.y*D],
                &pOutPDFGradsGPUPtr[neighIndex]);
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
        
    template<int D, int K, int U>
    KPBasis<D, K, U>::KPBasis(KPBasis::PointCorrelation ptCorr)
        :BasisInterface<D, K, U>(), ptCorr_(ptCorr)
    {
    }

    template<int D, int K, int U>
    KPBasis<D, K, U>::~KPBasis(void)
    {
    }

    template<int D, int K, int U>
    void KPBasis<D, K, U>::compute_basis_proj_pt_coords(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNeighbors,       
        const float* pInPtsGPUPtr,
        const float* pInSamplesGPUPtr,
        const float* pInInvRadiiGPUPtr,
        const int* pInNeighborsGPUPtr,
        const float* pInPDFsGPUPtr,
        const float* pInXNeighValsGPUPtr,
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
        if(ptCorr_ == KPBasis<D, K, U>::PointCorrelation::LINEAR){
            cFunct = (const void*)compute_kp_basis_proj_pt_coords<D, K, 0>;
        }else if(ptCorr_ == KPBasis<D, K, U>::PointCorrelation::GAUSS){
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
        if(ptCorr_ == KPBasis<D, K, U>::PointCorrelation::LINEAR){
            compute_kp_basis_proj_pt_coords<D, K, 0>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, 
                (const fpoint<D>*)pInPtsGPUPtr,
                (const fpoint<D>*)pInSamplesGPUPtr,
                (const fpoint<D>*)pInInvRadiiGPUPtr,
                (const int2*)pInNeighborsGPUPtr,
                pInPDFsGPUPtr, pInBasisGPUPtr, pOutProjGPUPtr);
        }else if(ptCorr_ == KPBasis<D, K, U>::PointCorrelation::GAUSS){
            compute_kp_basis_proj_pt_coords<D, K, 1>
                <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                pNumNeighbors, 
                (const fpoint<D>*)pInPtsGPUPtr,
                (const fpoint<D>*)pInSamplesGPUPtr,
                (const fpoint<D>*)pInInvRadiiGPUPtr,
                (const int2*)pInNeighborsGPUPtr,
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

    template<int D, int K, int U>
    void KPBasis<D, K, U>::compute_grads_basis_proj_pt_coords(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumNeighbors,       
        const float* pInPtsGPUPtr,
        const float* pInSamplesGPUPtr,
        const float* pInInvRadiiGPUPtr,
        const int* pInNeighborsGPUPtr,
        const float* pInPDFsGPUPtr,
        const float* pInXNeighValsGPUPtr,
        const float* pInBasisGPUPtr,
        const float* pInGradsGPUPtr,
        float* pOutBasisGradsGPUPtr,
        float* pOutPtsGradsGPUPtr,
        float* pOutSampleGradsGPUPtr,
        float* pOutPDFGradsGPUPtr,
        float* pOutXNeighGradsGPUPtr)
    {
        //Check if the gradietns of the points should be computed.
        bool pointGrads = (pOutPtsGradsGPUPtr != nullptr) &&
            (pOutSampleGradsGPUPtr != nullptr) &&
            (pOutPDFGradsGPUPtr != nullptr);
        
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
        if(ptCorr_ == KPBasis<D, K, U>::PointCorrelation::LINEAR){
            if(pointGrads)
                cFunct = (const void*)compute_kp_basis_proj_pt_coords_grads<D, K, 0, true>;
            else
                cFunct = (const void*)compute_kp_basis_proj_pt_coords_grads<D, K, 0, false>;
        }else if(ptCorr_ == KPBasis<D, K, U>::PointCorrelation::GAUSS){
            if(pointGrads)
                cFunct = (const void*)compute_kp_basis_proj_pt_coords_grads<D, K, 1, true>;
            else
                cFunct = (const void*)compute_kp_basis_proj_pt_coords_grads<D, K, 1, false>;
        }

#ifdef DEBUG_INFO
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, cudaStream);
#endif

        //Calculate the shared memory needed.
        unsigned int sharedMemSize = ((K + blockSize)*(D+1) + 
            (blockSize/K)*D + blockSize*(D*2 + 1))*sizeof(float);

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
        if(ptCorr_ == KPBasis<D, K, U>::PointCorrelation::LINEAR){
            if(pointGrads){
                compute_kp_basis_proj_pt_coords_grads<D, K, 0, true>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr);
            }else{
                compute_kp_basis_proj_pt_coords_grads<D, K, 0, false>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr);
            }
        }else if(ptCorr_ == KPBasis<D, K, U>::PointCorrelation::GAUSS){
            if(pointGrads){
                compute_kp_basis_proj_pt_coords_grads<D, K, 1, true>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr);
            }else{
                compute_kp_basis_proj_pt_coords_grads<D, K, 1, false>
                    <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
                    pNumNeighbors, pInPtsGPUPtr, pInSamplesGPUPtr, 
                    pInInvRadiiGPUPtr, (const int2*)pInNeighborsGPUPtr, pInPDFsGPUPtr, 
                    pInBasisGPUPtr, pInGradsGPUPtr, pOutBasisGradsGPUPtr, 
                    pOutPtsGradsGPUPtr, pOutSampleGradsGPUPtr, pOutPDFGradsGPUPtr);
            }
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
#define KP_BASIS_CLASS_DECL(D, K, U)    \
template class mccnn::KPBasis<D, K, U>;
DECLARE_TEMPLATE_DIMS_BASIS(KP_BASIS_CLASS_DECL)