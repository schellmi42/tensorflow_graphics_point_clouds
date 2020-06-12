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
/// \brief Class interface for a basis function projection.
/////////////////////////////////////////////////////////////////////////////

#ifndef BASIS_INTERFACE_CUH_
#define BASIS_INTERFACE_CUH_

#include "defines.hpp"
#include "gpu_device.hpp"

namespace mccnn{

    /**
     *  Interface for a basis function projection.
     *  @paramt D   Number of dimensions.
     *  @paramt K   Number of basis function used.
     *  @paramt U   Number of values per neighbor.
     */
    template<int D, int K, int U>
    class BasisInterface{

        public:

            /**
             *  Constructor.
             */
            BasisInterface(){}

            /**
             *  Destructor.
             */
            virtual ~BasisInterface(void){}

             /**
             *  Method to compute the projection of each point into a set
             *  of basis functions.
             *  @param  pDevice             Device object.
             *  @param  pNumNeighbors       Number of neighbors.
             *  @param  pInPtsGPUPtr        Input gpu pointer to the points.
             *  @param  pInSamplesGPUPtr    Input gpu pointer to the samples.
             *  @param  pInInvRadiiGPUPtr   Inverse radius of the receptive field.
             *  @param  pInNeighborsGPUPtr  Input gpu pointer to the neighbor list.
             *  @param  pInPDFsGPUPtr       Input gpu pointer to the pdf values.
             *  @param  pInXNeighValsGPUPtr Input gpu pointer to the per neighbor
             *      information.
             *  @param  pInBasisGPUPtr      Input gpu pointer to the kernel points.
             *  @param  pOutProjGPUPtr      Output pointer to the influences. 
             */
            virtual void compute_basis_proj_pt_coords(
                std::unique_ptr<IGPUDevice>& pDevice,
                const unsigned int pNumNeighbors,       
                const float* pInPtsGPUPtr,
                const float* pInSamplesGPUPtr,
                const float* pInInvRadiiGPUPtr,
                const int* pInNeighborsGPUPtr,
                const float* pInPDFsGPUPtr,
                const float* pInXNeighValsGPUPtr,
                const float* pInBasisGPUPtr,
                float* pOutProjGPUPtr) = 0;

            /**
             *  Method to compute the gradients of the projection of each point into a set
             *  of basis functions.
             *  @param  pDevice             Device object.
             *  @param  pNumNeighbors           Number of neighbors.
             *  @param  pInPtsGPUPtr            Input gpu pointer to the points.
             *  @param  pInSamplesGPUPtr        Input gpu pointer to the samples.
             *  @param  pInInvRadiiGPUPtr       Inverse radius of the receptive field.
             *  @param  pInNeighborsGPUPtr      Input gpu pointer to the neighbor list.
             *  @param  pInPDFsGPUPtr           Input gpu pointer to the pdf values.
             *  @param  pInXNeighValsGPUPtr Input gpu pointer to the per neighbor
             *      information.
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
                float* pOutXNeighGradsGPUPtr) = 0;
    };
}

#endif