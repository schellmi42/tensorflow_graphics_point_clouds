/////////////////////////////////////////////////////////////////////////////
/// \file basis_proj.cpp
///
/// \brief C++ operations definition to project input features to a set of 
///     basis functions.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "defines.hpp"
#include "tf_utils.hpp"
#include "tf_gpu_device.hpp"
#include "basis/basis_utils.cuh"
#include "basis/basis_proj.cuh"
#include "basis/basis_proj_grads.cuh"

/**
 *  Declaration of the tensorflow operations.
 */
REGISTER_OP("BasisProj")
    .Input("kernel_input: float32")
    .Input("pt_features: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("pdfs: float32")
    .Input("basis_func: float32")
    .Output("features: float32")
    .Attr("basis_type: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({
                pIC->Dim(pIC->input(3), 0), 
                pIC->Dim(pIC->input(1), 1),
                pIC->Dim(pIC->input(5), 0)});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

REGISTER_OP("BasisProjGrads")
    .Input("kernel_input: float32")
    .Input("pt_features: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("pdfs: float32")
    .Input("basis_func: float32")
    .Input("in_gradietns: float32")
    .Output("feat_gradients: float32")
    .Output("basis_gradients: float32")
    .Output("kernel_input_gradients: float32")
    .Output("pdf_gradients: float32")
    .Attr("basis_type: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        pIC->set_output(0, pIC->input(1));
        pIC->set_output(1, pIC->input(5));
        pIC->set_output(2, pIC->input(0));
        pIC->set_output(3, pIC->input(4));
        return Status::OK();
    });

namespace mccnn{

    /**
     *  Operation to project input features into a set of basis functions.
     */
    class BasisProjOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit BasisProjOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){

                OP_REQUIRES_OK(pContext, pContext->GetAttr("basis_type", &basisType_));
                OP_REQUIRES(pContext, basisType_ >= 0 && basisType_ < 6, 
                    errors::InvalidArgument("BasisProjOp requires a valid basis type."));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inKernelIns = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inNeighbors = pContext->input(2); 
                const Tensor& inSampleNeighIndices = pContext->input(3);
                const Tensor& inPDFs = pContext->input(4);
                const Tensor& inBasis = pContext->input(5);

                //Get variables from tensors.
                unsigned int numPts = inPtFeatures.shape().dim_size(0);
                unsigned int numSamples = inSampleNeighIndices.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inKernelIns.shape().dim_size(1);
                unsigned int numBasis = inBasis.shape().dim_size(0);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inKernelInGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inKernelIns);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* inPDFsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPDFs);
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasis);

                //Get the number of parameters of each basis function.
                unsigned int numParamsBasis = mccnn::get_num_params_x_basis(
                    (mccnn::BasisFunctType)basisType_, numDimensions);

                //Check for the correctness of the input.
                bool correct = false;
                for(int i = MIN_BASIS; i <= MAX_BASIS; i*=2)
                    correct = correct || (numBasis == i);
                OP_REQUIRES(pContext, correct, 
                    errors::InvalidArgument("BasisProjOp expects a valid number of basis functions."));
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("BasisProjOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, numInFeatures > 0 && (numInFeatures < 8 
                    || numInFeatures%MULTIPLE_IN_FEATURES==0), 
                    errors::InvalidArgument("BasisProjOp expects a valid number of input features, "
                    "between 1 and "+std::to_string(MULTIPLE_IN_FEATURES)+" or multiple of "
                    +std::to_string(MULTIPLE_IN_FEATURES)));
                OP_REQUIRES(pContext, inPtFeatures.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("BasisProjOp expects a number of points in"
                    " inPtFeatures equal to the number of points in the input points tensor"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("BasisProjOp expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));
                OP_REQUIRES(pContext, inSampleNeighIndices.shape().dim_size(0) == numSamples, 
                    errors::InvalidArgument("BasisProjOp expects the same number of samples "
                    "in inSampleNeighIndices as in the samples tensor."));
                OP_REQUIRES(pContext, inPDFs.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjOp expects a number of pdf values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inKernelIns.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjOp expects a number of kernel inputs values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inBasis.dims() == 2 && 
                    inBasis.shape().dim_size(1) == numParamsBasis, 
                    errors::InvalidArgument("BasisProjOp expects the rigth number of "
                    "parameters each for each basis function."));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{numSamples, numInFeatures, numBasis};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape, &outputGPUPtr));

                //Compute the convolution.
                DIMENSION_BASIS_SWITCH_CALL(numDimensions, numBasis, 
                    mccnn::basis_proj_gpu,
                    gpuDevice, (mccnn::BasisFunctType)basisType_, 
                    numSamples, numNeighbors, 
                    numInFeatures, inKernelInGPUPtr, inPtFeaturesGPUPtr, 
                    inNeighborsGPUPtr, inSampleNeighIGPUPtr, 
                    inBasisGPUPtr, inPDFsGPUPtr, outputGPUPtr)
            }

        private:

            /**Basis type.*/
            int   basisType_;
    };

    /**
     *  Operation to compute a monte carlo convolution.
     */
    class BasisProjGradsOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit BasisProjGradsOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){

                OP_REQUIRES_OK(pContext, pContext->GetAttr("basis_type", &basisType_));
                OP_REQUIRES(pContext, basisType_ >= 0 && basisType_ < 6, 
                    errors::InvalidArgument("BasisProjGradsOp requires a valid basis type."));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inKernelIns = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inNeighbors = pContext->input(2); 
                const Tensor& inSampleNeighIndices = pContext->input(3);
                const Tensor& inPDFs = pContext->input(4);
                const Tensor& inBasis = pContext->input(5);
                const Tensor& inGradients = pContext->input(6);

                //Get variables from tensors.
                unsigned int numPts = inPtFeatures.shape().dim_size(0);
                unsigned int numSamples = inSampleNeighIndices.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inKernelIns.shape().dim_size(1);
                unsigned int numBasis = inBasis.shape().dim_size(0);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inKernelInsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inKernelIns);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* inPDFsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPDFs);
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasis);
                const float* inGradientsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inGradients);

                //Get the number of parameters of each basis function.
                unsigned int numParamsBasis = mccnn::get_num_params_x_basis(
                    (mccnn::BasisFunctType)basisType_, numDimensions);

                //Check for the correctness of the input.
                bool correct = false;
                for(int i = MIN_BASIS; i <= MAX_BASIS; i*=2)
                    correct = correct || (numBasis == i);
                OP_REQUIRES(pContext, correct, 
                    errors::InvalidArgument("BasisProjGradsOp expects a valid number of basis functions."));
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("BasisProjGradsOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, numInFeatures > 0 && (numInFeatures < 8 
                    || numInFeatures%MULTIPLE_IN_FEATURES==0), 
                    errors::InvalidArgument("BasisProjGradsOp expects a valid number of input features, "
                    "between 1 and "+std::to_string(MULTIPLE_IN_FEATURES)+" or multiple of "
                    +std::to_string(MULTIPLE_IN_FEATURES)));
                OP_REQUIRES(pContext, inPtFeatures.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("BasisProjGradsOp expects a number of points in"
                    " inPtFeatures equal to the number of points in the input points tensor"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("BasisProjGradsOp expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));
                OP_REQUIRES(pContext, inSampleNeighIndices.shape().dim_size(0) == numSamples, 
                    errors::InvalidArgument("BasisProjGradsOp expects the same number of points "
                    "in inSampleNeighIndices as in the samples tensor."));
                OP_REQUIRES(pContext, inPDFs.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjGradsOp expects a number of pdf values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inKernelIns.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjOp expects a number of kernel inputs values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inBasis.dims() == 2 && 
                    inBasis.shape().dim_size(1) == numParamsBasis, 
                    errors::InvalidArgument("BasisProjGradsOp expects the rigth number of "
                    "parameters each for each basis function."));
                OP_REQUIRES(pContext, inGradients.dims() == 3 && 
                    inGradients.shape().dim_size(0) == numSamples &&
                    inGradients.shape().dim_size(1) == numInFeatures &&
                    inGradients.shape().dim_size(2) == numBasis, 
                    errors::InvalidArgument("BasisProjGradsOp expects the rigth number of feaure gradients,"));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* output1GPUPtr = nullptr;
                float* output2GPUPtr = nullptr;
                float* output3GPUPtr = nullptr;
                float* output4GPUPtr = nullptr;
                TensorShape outShape1 = TensorShape{numPts, numInFeatures};
                TensorShape outShape2 = TensorShape{numBasis, numParamsBasis};
                TensorShape outShape3 = TensorShape{numNeighbors, numDimensions};
                TensorShape outShape4 = TensorShape{numNeighbors};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape1, &output1GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (1, pContext, outShape2, &output2GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (2, pContext, outShape3, &output3GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (3, pContext, outShape4, &output4GPUPtr));

                //Compute the convolution gradients.
                DIMENSION_BASIS_SWITCH_CALL(numDimensions, numBasis, 
                    mccnn::basis_proj_grads_gpu,
                    gpuDevice, (mccnn::BasisFunctType)basisType_, 
                    numPts, numSamples, numNeighbors, 
                    numInFeatures, inKernelInsGPUPtr, inPtFeaturesGPUPtr, 
                    inNeighborsGPUPtr, inSampleNeighIGPUPtr, inBasisGPUPtr, 
                    inPDFsGPUPtr, inGradientsGPUPtr, output1GPUPtr, output2GPUPtr,
                    output3GPUPtr, output4GPUPtr)
            }

        private:

            /**Basis type.*/
            int   basisType_;
    };
}

REGISTER_KERNEL_BUILDER(Name("BasisProj").Device(DEVICE_GPU), mccnn::BasisProjOp);
REGISTER_KERNEL_BUILDER(Name("BasisProjGrads").Device(DEVICE_GPU), mccnn::BasisProjGradsOp);