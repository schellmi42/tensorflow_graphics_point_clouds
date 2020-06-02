/////////////////////////////////////////////////////////////////////////////
/// \file basis_utils.cuh
///
/// \brief Basis utils file.
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
#include "basis/basis_utils.cuh"
//Include different basis functions.
#include "basis/basis_kp.cuh"
#include "basis/basis_hproj.cuh"
#include "basis/basis_hproj_bilateral.cuh"

template<int D, int K, int U>
std::unique_ptr<mccnn::BasisInterface<D, K, U>> 
mccnn::basis_function_factory(mccnn::BasisFunctType pBasisType)
{
    if(pBasisType == mccnn::BasisFunctType::KERNEL_POINT_LINEAR){
        return make_unique<mccnn::KPBasis<D, K, U>>(
            mccnn::KPBasis<D, K, U>::PointCorrelation::LINEAR);
    }else if(pBasisType == mccnn::BasisFunctType::KERNEL_POINT_GAUSS){
        return make_unique<mccnn::KPBasis<D, K, U>>(
            mccnn::KPBasis<D, K, U>::PointCorrelation::GAUSS);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_RELU){
        return make_unique<mccnn::HProjBasis<D, K, U>>(
            mccnn::HProjBasis<D, K, U>::ActivationFunction::RELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_LRELU){
        return make_unique<mccnn::HProjBasis<D, K, U>>(
            mccnn::HProjBasis<D, K, U>::ActivationFunction::LRELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_ELU){
        return make_unique<mccnn::HProjBasis<D, K, U>>(
            mccnn::HProjBasis<D, K, U>::ActivationFunction::ELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_EXP){
        return make_unique<mccnn::HProjBasis<D, K, U>>(
            mccnn::HProjBasis<D, K, U>::ActivationFunction::EXP);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_BIL_RELU){
        return make_unique<mccnn::HProjBilateralBasis<D, K, U>>(
            mccnn::HProjBilateralBasis<D, K, U>::ActivationFunction::RELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_BIL_LRELU){
        return make_unique<mccnn::HProjBilateralBasis<D, K, U>>(
            mccnn::HProjBilateralBasis<D, K, U>::ActivationFunction::LRELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_BIL_ELU){
        return make_unique<mccnn::HProjBilateralBasis<D, K, U>>(
            mccnn::HProjBilateralBasis<D, K, U>::ActivationFunction::ELU);
    }else if(pBasisType == mccnn::BasisFunctType::HPROJ_BIL_EXP){
        return make_unique<mccnn::HProjBilateralBasis<D, K, U>>(
            mccnn::HProjBilateralBasis<D, K, U>::ActivationFunction::EXP);
    }
    return std::unique_ptr<mccnn::BasisInterface<D, K, U>>(nullptr);
}

#define BASIS_FUNCTION_FACTORY_DECL(D, K, U)                   \
    template std::unique_ptr<mccnn::BasisInterface<D, K, U>>   \
    mccnn::basis_function_factory<D, K, U>                     \
    (mccnn::BasisFunctType pBasisType);

DECLARE_TEMPLATE_DIMS_BASIS(BASIS_FUNCTION_FACTORY_DECL)