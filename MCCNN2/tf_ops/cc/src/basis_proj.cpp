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
    .Input("points: float32")
    .Input("pt_features: float32")
    .Input("samples: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("inv_radii: float32")
    .Input("pdfs: float32")
    .Input("basis_func: float32")
    .Output("features: float32")
    .Attr("basis_type: int")
    .Attr("pt_grads: bool")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({
                pIC->Dim(pIC->input(2), 0), 
                pIC->Dim(pIC->input(1), 1),
                pIC->Dim(pIC->input(7), 0)});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

REGISTER_OP("BasisProjGrads")
    .Input("points: float32")
    .Input("pt_features: float32")
    .Input("samples: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("inv_radii: float32")
    .Input("pdfs: float32")
    .Input("basis_func: float32")
    .Input("in_gradietns: float32")
    .Output("feat_gradients: float32")
    .Output("basis_gradients: float32")
    .Attr("basis_type: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        pIC->set_output(0, pIC->input(1));
        pIC->set_output(1, pIC->input(7));
        return Status::OK();
    });

REGISTER_OP("BasisProjGradsWithPtGrads")
    .Input("points: float32")
    .Input("pt_features: float32")
    .Input("samples: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("inv_radii: float32")
    .Input("pdfs: float32")
    .Input("basis_func: float32")
    .Input("in_gradietns: float32")
    .Output("feat_gradients: float32")
    .Output("basis_gradients: float32")
    .Output("point_gradients: float32")
    .Output("sample_gradients: float32")
    .Output("pdf_gradients: float32")
    .Attr("basis_type: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        pIC->set_output(0, pIC->input(1));
        pIC->set_output(1, pIC->input(7));
        pIC->set_output(2, pIC->input(0));
        pIC->set_output(3, pIC->input(2));
        pIC->set_output(4, pIC->input(6));
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
                const Tensor& inPts = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inSamples = pContext->input(2); 
                const Tensor& inNeighbors = pContext->input(3); 
                const Tensor& inSampleNeighIndices = pContext->input(4);
                const Tensor& inInvRadii = pContext->input(5);
                const Tensor& inPDFs = pContext->input(6);
                const Tensor& inBasis = pContext->input(7);

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numSamples = inSamples.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);
                unsigned int numBasis = inBasis.shape().dim_size(0);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inPtsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const float* inSamplesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inSamples);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* inInvRadiiGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvRadii);
                const float* inPDFsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPDFs);
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasis);

                //Get the number of parameters of each basis function.
                unsigned int numParamsBasis = mccnn::get_num_params_x_basis(
                    (mccnn::BasisFunctType)basisType_, numDimensions, 0);

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
                OP_REQUIRES(pContext, inSamples.shape().dim_size(1) == numDimensions, 
                    errors::InvalidArgument("BasisProjOp expects a number of dimensions in"
                    " inSamples equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvRadii.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("BasisProjOp expects a number of dimensions in"
                    " inInvRadii equal to the number of dimensions in the input points"));
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
                int numXNeighVals = 0;
                DIMENSION_BASIS_SWITCH_CALL(numDimensions, numBasis, numXNeighVals, 
                    mccnn::basis_proj_gpu,
                    gpuDevice, (mccnn::BasisFunctType)basisType_, numSamples, numNeighbors, 
                    numInFeatures, inPtsGPUPtr, inPtFeaturesGPUPtr, inSamplesGPUPtr, 
                    inNeighborsGPUPtr, inSampleNeighIGPUPtr, inInvRadiiGPUPtr, 
                    inBasisGPUPtr, inPDFsGPUPtr, nullptr, outputGPUPtr)
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
                const Tensor& inPts = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inSamples = pContext->input(2); 
                const Tensor& inNeighbors = pContext->input(3); 
                const Tensor& inSampleNeighIndices = pContext->input(4);
                const Tensor& inInvRadii = pContext->input(5);
                const Tensor& inPDFs = pContext->input(6);
                const Tensor& inBasis = pContext->input(7);
                const Tensor& inGradients = pContext->input(8);

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numSamples = inSamples.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);
                unsigned int numBasis = inBasis.shape().dim_size(0);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inPtsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const float* inSamplesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inSamples);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* inInvRadiiGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvRadii);
                const float* inPDFsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPDFs);
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasis);
                const float* inGradientsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inGradients);

                //Get the number of parameters of each basis function.
                unsigned int numParamsBasis = mccnn::get_num_params_x_basis(
                    (mccnn::BasisFunctType)basisType_, numDimensions, 0);

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
                OP_REQUIRES(pContext, inSamples.shape().dim_size(1) == numDimensions, 
                    errors::InvalidArgument("BasisProjGradsOp expects a number of dimensions in"
                    " inSamples equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvRadii.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("BasisProjGradsOp expects a number of dimensions in"
                    " inInvRadii equal to the number of dimensions in the input points"));
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
                TensorShape outShape1 = TensorShape{numPts, numInFeatures};
                TensorShape outShape2 = TensorShape{numBasis, numParamsBasis};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape1, &output1GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (1, pContext, outShape2, &output2GPUPtr));

                //Compute the convolution gradients.
                int numXNeighVals = 0;
                DIMENSION_BASIS_SWITCH_CALL(numDimensions, numBasis, numXNeighVals, 
                    mccnn::basis_proj_grads_gpu,
                    gpuDevice, (mccnn::BasisFunctType)basisType_, numPts, numSamples, numNeighbors, 
                    numInFeatures, inPtsGPUPtr, inPtFeaturesGPUPtr, inSamplesGPUPtr, 
                    inNeighborsGPUPtr, inSampleNeighIGPUPtr, inInvRadiiGPUPtr, inBasisGPUPtr, 
                    inPDFsGPUPtr, nullptr, inGradientsGPUPtr, output1GPUPtr, output2GPUPtr,
                    nullptr, nullptr, nullptr, nullptr)
            }

        private:

            /**Basis type.*/
            int   basisType_;
    };

    /**
     *  Operation to compute a monte carlo convolution.
     */
    class BasisProjGradsWithPtGradsOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit BasisProjGradsWithPtGradsOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){

                OP_REQUIRES_OK(pContext, pContext->GetAttr("basis_type", &basisType_));
                OP_REQUIRES(pContext, basisType_ >= 0 && basisType_ < 6, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp requires a valid basis type."));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inPts = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inSamples = pContext->input(2); 
                const Tensor& inNeighbors = pContext->input(3); 
                const Tensor& inSampleNeighIndices = pContext->input(4);
                const Tensor& inInvRadii = pContext->input(5);
                const Tensor& inPDFs = pContext->input(6);
                const Tensor& inBasis = pContext->input(7);
                const Tensor& inGradients = pContext->input(8);

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numSamples = inSamples.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);
                unsigned int numBasis = inBasis.shape().dim_size(0);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inPtsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const float* inSamplesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inSamples);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* inInvRadiiGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvRadii);
                const float* inPDFsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPDFs);
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasis);
                const float* inGradientsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inGradients);

                //Get the number of parameters of each basis function.
                unsigned int numParamsBasis = mccnn::get_num_params_x_basis(
                    (mccnn::BasisFunctType)basisType_, numDimensions, 0);

                //Check for the correctness of the input.
                bool correct = false;
                for(int i = MIN_BASIS; i <= MAX_BASIS; i*=2)
                    correct = correct || (numBasis == i);
                OP_REQUIRES(pContext, correct, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects a valid number of basis functions."));
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, numInFeatures > 0 && (numInFeatures < 8 
                    || numInFeatures%MULTIPLE_IN_FEATURES==0), 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects a valid number of input features, "
                    "between 1 and "+std::to_string(MULTIPLE_IN_FEATURES)+" or multiple of "
                    +std::to_string(MULTIPLE_IN_FEATURES)));
                OP_REQUIRES(pContext, inSamples.shape().dim_size(1) == numDimensions, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects a number of dimensions in"
                    " inSamples equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvRadii.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects a number of dimensions in"
                    " inInvRadii equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inPtFeatures.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects a number of points in"
                    " inPtFeatures equal to the number of points in the input points tensor"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));
                OP_REQUIRES(pContext, inSampleNeighIndices.shape().dim_size(0) == numSamples, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects the same number of points "
                    "in inSampleNeighIndices as in the samples tensor."));
                OP_REQUIRES(pContext, inPDFs.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects a number of pdf values equal "
                    "to the number of neighbors."));
                OP_REQUIRES(pContext, inBasis.dims() == 2 && 
                    inBasis.shape().dim_size(1) == numParamsBasis, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects the rigth number of "
                    "parameters each for each basis function."));
                OP_REQUIRES(pContext, inGradients.dims() == 3 && 
                    inGradients.shape().dim_size(0) == numSamples &&
                    inGradients.shape().dim_size(1) == numInFeatures &&
                    inGradients.shape().dim_size(2) == numBasis, 
                    errors::InvalidArgument("BasisProjGradsWithPtGradsOp expects the rigth number of feaure gradients,"));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* output1GPUPtr = nullptr;
                float* output2GPUPtr = nullptr;
                float* output3GPUPtr = nullptr;
                float* output4GPUPtr = nullptr;
                float* output5GPUPtr = nullptr;
                TensorShape outShape1 = TensorShape{numPts, numInFeatures};
                TensorShape outShape2 = TensorShape{numBasis, numParamsBasis};
                TensorShape outShape3 = TensorShape{numPts, numDimensions};
                TensorShape outShape4 = TensorShape{numSamples, numDimensions};
                TensorShape outShape5 = TensorShape{numNeighbors};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape1, &output1GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (1, pContext, outShape2, &output2GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (2, pContext, outShape3, &output3GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (3, pContext, outShape4, &output4GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (4, pContext, outShape5, &output5GPUPtr));

                //Compute the convolution gradients.
                int numXNeighVals = 0;
                DIMENSION_BASIS_SWITCH_CALL(numDimensions, numBasis, numXNeighVals, 
                    mccnn::basis_proj_grads_gpu,
                    gpuDevice, (mccnn::BasisFunctType)basisType_, numPts, numSamples, numNeighbors, 
                    numInFeatures, inPtsGPUPtr, inPtFeaturesGPUPtr, inSamplesGPUPtr, 
                    inNeighborsGPUPtr, inSampleNeighIGPUPtr, inInvRadiiGPUPtr, inBasisGPUPtr, 
                    inPDFsGPUPtr, nullptr, inGradientsGPUPtr, output1GPUPtr, output2GPUPtr,
                    output3GPUPtr, output4GPUPtr, output5GPUPtr, nullptr)
            }

        private:

            /**Basis type.*/
            int   basisType_;
    };
}

REGISTER_KERNEL_BUILDER(Name("BasisProj").Device(DEVICE_GPU), mccnn::BasisProjOp);
REGISTER_KERNEL_BUILDER(Name("BasisProjGrads").Device(DEVICE_GPU), mccnn::BasisProjGradsOp);
REGISTER_KERNEL_BUILDER(Name("BasisProjGradsWithPtGrads").Device(DEVICE_GPU), mccnn::BasisProjGradsWithPtGradsOp);