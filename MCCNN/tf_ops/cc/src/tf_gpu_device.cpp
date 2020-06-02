/////////////////////////////////////////////////////////////////////////////
/// \file tf_gpu_device.cpp
///
/// \brief Implementation of the tensorflow gpu device.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "cuda_runtime.h"

#include "tf_gpu_device.hpp"
#include "tf_utils.hpp"

#include <stdio.h>
#include <stdlib.h>

namespace mccnn{

    TFGPUDevice::TFGPUDevice(tensorflow::OpKernelContext* pContext):
        IGPUDevice(), context_(pContext)
    {
        //Get GPU device.
        cudaDeviceProp prop;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        
        deviceProps_.warpSize_ = prop.warpSize;
        deviceProps_.numMPs_ = prop.multiProcessorCount;
        deviceProps_.maxThreadsXBlock_ = prop.maxThreadsPerBlock;
        deviceProps_.maxThreadsXMP_ = prop.maxThreadsPerMultiProcessor;
        deviceProps_.maxRegistersXBlock_ = prop.regsPerBlock;
        deviceProps_.maxRegistersXMP_ = prop.regsPerMultiprocessor;
        deviceProps_.sharedMemXBlock_ = prop.sharedMemPerBlock;
        deviceProps_.sharedMemXMP_ = prop.sharedMemPerMultiprocessor;
        deviceProps_.majorVersion_ = prop.major;
        deviceProps_.minorVersion_ = prop.minor;

#ifdef DEBUG_INFO
        static int firstPrint = 0;
        if(firstPrint == 0){
            firstPrint = 1;
            fprintf(stderr, "### GPU INFO ###\n");
            fprintf(stderr, "Warp size: %d\n", deviceProps_.warpSize_);
            fprintf(stderr, "Num MP: %d\n", deviceProps_.numMPs_);
            fprintf(stderr, "Max threads x block: %d\n", deviceProps_.maxThreadsXBlock_);
            fprintf(stderr, "Max threads x MP: %d\n", deviceProps_.maxThreadsXMP_);
            fprintf(stderr, "Max registers x block: %d\n", deviceProps_.maxRegistersXBlock_);
            fprintf(stderr, "Max registers x MP: %d\n", deviceProps_.maxRegistersXMP_);
            fprintf(stderr, "Max shared mem x block: %d\n", (int)deviceProps_.sharedMemXBlock_);
            fprintf(stderr, "Max shared mem x MP: %d\n", (int)deviceProps_.sharedMemXMP_);
            fprintf(stderr, "Compute version: %d.%d\n", deviceProps_.majorVersion_, deviceProps_.minorVersion_);
            fprintf(stderr, "\n");
        }
#endif
    }

    TFGPUDevice::~TFGPUDevice()
    {}

    void TFGPUDevice::memset(void* pDest, int pVal, size_t pSize)
    {
        cudaMemsetAsync(pDest, pVal, pSize, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    void TFGPUDevice::memcpy_device_to_device(void* pDest, void* pSrc, size_t pSize)
    {
        cudaMemcpyAsync(pDest, pSrc, pSize, cudaMemcpyDeviceToDevice, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    void TFGPUDevice::memcpy_device_to_host(void* pDest, void* pSrc, size_t pSize)
    {
        cudaMemcpyAsync(pDest, pSrc, pSize, cudaMemcpyDeviceToHost, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    void TFGPUDevice::memcpy_host_to_device(void* pDest, void* pSrc, size_t pSize)
    {
        cudaMemcpyAsync(pDest, pSrc, pSize, cudaMemcpyHostToDevice, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    void TFGPUDevice::memcpy_host_to_symbol(void* pDest, void* pSrc, size_t pSize)
    {
        cudaMemcpyToSymbolAsync(pDest, pSrc, pSize, 0, cudaMemcpyHostToDevice, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    int TFGPUDevice::get_max_active_block_x_sm(
                const unsigned int pBlockSize, 
                const void* pFunct,
                const size_t pSharedMemXBlock)
    {
        int outputNumBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor ( 
            &outputNumBlocks, pFunct, pBlockSize, pSharedMemXBlock);
        return outputNumBlocks;
    }

    void TFGPUDevice::check_error(
        const char* pFile, 
        int pLine)
    {
        cudaError_t errorCode = cudaPeekAtLastError();
        if (errorCode != cudaSuccess) 
        {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(errorCode), pFile, pLine);
            exit(errorCode);
            //TODO - Proper error handling, exceptions.
        }
    }

    float* TFGPUDevice::getFloatTmpGPUBuffer(const unsigned int pSize)
    {
        return this->getTmpGPUBuffer<float>(pSize);
    }

    int* TFGPUDevice::getIntTmpGPUBuffer(const unsigned int pSize)
    {
        return this->getTmpGPUBuffer<int>(pSize);
    }

    int64_m* TFGPUDevice::getInt64TmpGPUBuffer(const unsigned int pSize)
    {
        return this->getTmpGPUBuffer<mccnn::int64_m>(pSize);
    }

    const cudaStream_t& TFGPUDevice::getCUDAStream()
    {
        return context_->eigen_device<Eigen::GpuDevice>().stream();
    }

    template<class T>
    T* TFGPUDevice::getTmpGPUBuffer(const unsigned int pSize)
    {
        std::unique_ptr<tensorflow::Tensor> pTmpTensor = make_unique<tensorflow::Tensor>();
        TensorShape tmpShape = TensorShape{pSize};
        if(!TF_PREDICT_TRUE(context_->allocate_temp(
            DataTypeToEnum<T>::value, tmpShape, pTmpTensor.get()).ok())){
            fprintf(stderr,"Error allocating temporal tensor of %ld bytes.\n", sizeof(T)*pSize);
            exit(-1);
            //TODO - Proper error handling, exceptions.
        }
        auto tmpTensorFlat = pTmpTensor->flat<T>();
        tmpTensors_.push_back(std::move(pTmpTensor));
        return &(tmpTensorFlat(0));
    }

    template int* TFGPUDevice::getTmpGPUBuffer<int>(const unsigned int pSize);
    template float* TFGPUDevice::getTmpGPUBuffer<float>(const unsigned int pSize);

}