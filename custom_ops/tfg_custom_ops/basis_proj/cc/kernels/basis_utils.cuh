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

#ifndef BASIS_UTILS_CUH_
#define BASIS_UTILS_CUH_

#include "defines.hpp"
#include "math_helper.cuh"
#include "basis/basis_interface.cuh"

//Definition of the minimum and maximum kernel points.
#define MIN_BASIS 8
#define MAX_BASIS 32

//Definition of the number of which the number of features should be 
// multiple of.
#define MULTIPLE_IN_FEATURES 8

//Macros to declare and call a template function with a variable
//number of dimensions and variable basis functions.
#define DECLARE_TEMPLATE_DIMS_BASIS(Func)  \
    Func(2, 8 )                         \
    Func(2, 16)                         \
    Func(2, 32)                         \
    Func(3, 8 )                         \
    Func(3, 16)                         \
    Func(3, 32)                         \
    Func(4, 8 )                         \
    Func(4, 16)                         \
    Func(4, 32)                         \
    Func(5, 8 )                         \
    Func(5, 16)                         \
    Func(5, 32)                         \
    Func(6, 8 )                         \
    Func(6, 16)                         \
    Func(6, 32)                         


#define BASIS_CASE_SWITCH(Dim, K, Func, ...)                   \
    case K:                                                    \
        Func<Dim, K>(__VA_ARGS__);                             \
        break;

#define DIM_CASE_BASIS_SWITCH_CALL(Dim, Var, Func, ...)             \
    case Dim:                                                       \
        switch(Var){                                                \
            BASIS_CASE_SWITCH(Dim, 8, Func, __VA_ARGS__)            \
            BASIS_CASE_SWITCH(Dim, 16, Func, __VA_ARGS__)           \
            BASIS_CASE_SWITCH(Dim, 32, Func, __VA_ARGS__)           \
        };                                                          \
        break;

#define DIMENSION_BASIS_SWITCH_CALL(Var1, Var2, Func, ...)              \
    switch(Var1){                                                       \
        DIM_CASE_BASIS_SWITCH_CALL(2, Var2, Func, __VA_ARGS__)          \
        DIM_CASE_BASIS_SWITCH_CALL(3, Var2, Func, __VA_ARGS__)          \
        DIM_CASE_BASIS_SWITCH_CALL(4, Var2, Func, __VA_ARGS__)          \
        DIM_CASE_BASIS_SWITCH_CALL(5, Var2, Func, __VA_ARGS__)          \
        DIM_CASE_BASIS_SWITCH_CALL(6, Var2, Func, __VA_ARGS__)          \
    };

namespace mccnn{

    /**
     *  Types of basis functions available.
     */
    enum class BasisFunctType : int { 
        KERNEL_POINT_LINEAR=0,
        KERNEL_POINT_GAUSS=1,
        HPROJ_RELU=2,
        HPROJ_LRELU=3,
        HPROJ_ELU=4,
        HPROJ_EXP=5
    };

    /**
     *  Method to get the number of parameters of each basis function.
     *  @param  pType       Type of basis function.
     *  @param  pDimensions Number of dimensions.
     *  @param  pXNeighVals Number of values per neighbors.
     *  @return Number of parameters of each basis function.
     */
    __forceinline__ unsigned int get_num_params_x_basis(
        BasisFunctType pType,
        const int pDimensions)
    {
        unsigned int result = 0;
        switch(pType)
        {
            case BasisFunctType::KERNEL_POINT_LINEAR:
                result = pDimensions+1;
                break;
            case BasisFunctType::KERNEL_POINT_GAUSS:
                result = pDimensions+1;
                break;
            case BasisFunctType::HPROJ_RELU:
                result = pDimensions+1;
                break;
            case BasisFunctType::HPROJ_LRELU:
                result = pDimensions+1;
                break;
            case BasisFunctType::HPROJ_ELU:
                result = pDimensions+1;
                break;
            case BasisFunctType::HPROJ_EXP:
                result = pDimensions+1;
                break;
        }
        return result;
    }

    /**
     *  Method to create an object of a basis projector.
     *  @param  pBasisType  Type of basis function used.
     *  @return Basis projector object.
     */
    template<int D, int K>
    std::unique_ptr<BasisInterface<D, K>> 
    basis_function_factory(BasisFunctType pBasisType);
}

#endif