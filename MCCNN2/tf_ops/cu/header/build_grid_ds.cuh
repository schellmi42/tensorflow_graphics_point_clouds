/////////////////////////////////////////////////////////////////////////////
/// \file build_grid_ds.cuh
///
/// \brief Declaraion of the CUDA operations to build the data structure to 
///     access the sparse regular grid. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef BUILD_GRID_DS_CUH_
#define BUILD_GRID_DS_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to compute the data structure to access a sparse regular grid.
     *  @param  pDevice                 Device.
     *  @param  pDSSize                 Size of the data structure.
     *  @param  pNumPts                 Number of points.
     *  @param  pInGPUPtrKeys           Input pointer to the vector of keys 
     *      on the GPU.
     *  @param  pInGPUPtrNumCells       Input pointer to the vector of number  
     *      of cells on the GPU.
     *  @param  pInGPUpOutGPUPtrKeys    Output pointer to the data structure  
     *      on the GPU.
     *  @paramT D                       Number of dimensions.
     */
    template<int D>
    void build_grid_ds_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pDSSize, 
        const unsigned int pNumPts,
        const mccnn::int64_m* pInGPUPtrKeys,
        const int* pInGPUPtrNumCells,
        int* pOutGPUPtrDS);
}

#endif