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
/// \brief Basis utils file.
/////////////////////////////////////////////////////////////////////////////

#include "defines.hpp"
#include "cuda_kernel_utils.cuh"
#include "math_helper.cuh"
#include "basis/basis_utils.cuh"
//Include different basis functions.
#include "basis/basis_kp.cuh"
#include "basis/basis_hproj.cuh"

template<int D, int K>
std::unique_ptr<mccnn::BasisInterface<D, K>> 
mccnn::basis_function_factory(mccnn::BasisFunctType pBasisType)
{
    if(pBasisType == mccnn::BasisFunctType::KERNEL_POINT_LINEAR){
        return make_unique<mccnn::KPBasis<D, K>>(
            mccnn::KPBasis<D, K>::PointCorrelation::LINEAR);
    }else if(pBasisType == mccnn::BasisFunctType::KERNEL_POINT_GAUSS){
        return make_unique<mccnn::KPBasis<D, K>>(
            mccnn::KPBasis<D, K>::PointCorrelation::GAUSS);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_RELU){
        return make_unique<mccnn::HProjBasis<D, K>>(
            mccnn::HProjBasis<D, K>::ActivationFunction::RELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_LRELU){
        return make_unique<mccnn::HProjBasis<D, K>>(
            mccnn::HProjBasis<D, K>::ActivationFunction::LRELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_ELU){
        return make_unique<mccnn::HProjBasis<D, K>>(
            mccnn::HProjBasis<D, K>::ActivationFunction::ELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_EXP){
        return make_unique<mccnn::HProjBasis<D, K>>(
            mccnn::HProjBasis<D, K>::ActivationFunction::EXP);
    }
    return std::unique_ptr<mccnn::BasisInterface<D, K>>(nullptr);
}

#define BASIS_FUNCTION_FACTORY_DECL(D, K)                   \
    template std::unique_ptr<mccnn::BasisInterface<D, K>>   \
    mccnn::basis_function_factory<D, K>                     \
    (mccnn::BasisFunctType pBasisType);

DECLARE_TEMPLATE_DIMS_BASIS(BASIS_FUNCTION_FACTORY_DECL)