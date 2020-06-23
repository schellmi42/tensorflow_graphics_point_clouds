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

#ifndef BASIS_KP_CUH_
#define BASIS_KP_CUH_

#include "defines.hpp"
#include "basis_interface.cuh"

namespace mccnn{

    /**
     *  Kernel points basis function projection.
     *  @paramt D   Number of dimensions.
     *  @paramt K   Number of basis function used.
     */
    template<int D, int K>
    class KPBasis: public BasisInterface<D, K>{
 
        public:

            /**
             *  Point correlation.
             */
            enum class PointCorrelation : int { 
                LINEAR=0,
                GAUSS=1
            };
 
            /**
             *  Constructor.
             *  @param  ptCorr  Point correlation.
             */
            KPBasis(PointCorrelation ptCorr);
 
            /**
             *  Destructor.
             */
            virtual ~KPBasis(void);

            /**
             *  Method to compute the projection of each point into a set
             *  of basis functions.
             *  @param  pDevice             Device object.
             *  @param  pNumNeighbors       Number of neighbors.
             *  @param  pInKernelInGPUPtr   Input gpu pointer to the kernel inputs.
             *  @param  pInPDFsGPUPtr       Input gpu pointer to the pdf values.
             *  @param  pInBasisGPUPtr      Input gpu pointer to the kernel points.
             *  @param  pOutProjGPUPtr      Output pointer to the influences. 
             */
             virtual void compute_basis_proj_pt_coords(
                std::unique_ptr<IGPUDevice>& pDevice,
                const unsigned int pNumNeighbors,       
                const float* pInKernelInGPUPtr,
                const float* pInPDFsGPUPtr,
                const float* pInBasisGPUPtr,
                float* pOutProjGPUPtr);

            /**
             *  Method to compute the gradients of the projection of each point into a set
             *  of basis functions.
             *  @param  pDevice             Device object.
             *  @param  pNumNeighbors           Number of neighbors.
             *  @param  pInKernelInGPUPtr       Input gpu pointer to the kernel inputs.
             *  @param  pInPDFsGPUPtr           Input gpu pointer to the pdf values.
             *  @param  pInBasisGPUPtr          Input gpu pointer to the kernel points.
             *  @param  pInGradsGPUPtr          Input gpu pointer to the input gradients.
             *  @param  pOutBasisGradsGPUPtr    Output pointer to the basis gradients. 
             *  @param  pOutPtsGradsGPUPtr      Output pointer to the gradients of
             *      the pointers.
             *  @param  pOutSampleGradsGPUPtr  Output pointer to the gradients of
             *      the samples.
             *  @param  pOutPDFGradsGPUPtr     Output pointer to the gradients of
             *      the pdfs.
             *  @param  pOutXNeighGradsGPUPtr  Output pointer to the gradients of
             *      the pdfs.
             */
             virtual void compute_grads_basis_proj_pt_coords(
                std::unique_ptr<IGPUDevice>& pDevice,
                const unsigned int pNumNeighbors,       
                const float* pInKernelInGPUPtr,
                const float* pInPDFsGPUPtr,
                const float* pInBasisGPUPtr,
                const float* pInGradsGPUPtr,
                float* pOutBasisGradsGPUPtr,
                float* pOutKernelInsGradsGPUPtr,
                float* pOutPDFGradsGPUPtr);

        private:

            /**Point correlation method used.*/
            PointCorrelation ptCorr_;
    };
}

#endif