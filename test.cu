#include <iostream>
#include <sstream>
#include <chrono>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#define CHK_CUTLASS(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CHK_CU(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassSimt;

// CUDA SM architecture number
using SmArch = cutlass::arch::Sm86;

// the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;  // Threadblock tile shape

// tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;         // Warp tile shape

// the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;    // TensorCore instruction shape

// how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 2;

// the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    1,                                                 // The number of elements per vectorized.
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination


using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Parameters {

  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  Parameters() {}
  Parameters(int n, int w, int h, int c, int k, int k_w, int k_h):
    input_size(n, w, h, c),
    filter_size(k, k_w, k_h, c),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    alpha(1),
    beta(0) { }

  // Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    return cutlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  // Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  const int N = 1;
  Parameters paras(1, 224, 224, 3, 64, 3, 3);
  
  // Define arguments for CUTLASS Convolution
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation; // mode (kCrossCorrelation or kConvolution)
  int split_k_slices = 1; // Split K dimension into 1 partitions

  // Allocate host-device tensors using the CUTLASS Utilities.
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(paras.input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(paras.filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(paras.output_size());
 
  // Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(7),
      ElementInputA(-8),
      0);

  // Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8),
      0);

  // Fill tensor C on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_c.host_view());

    // Construct Conv2dProblemSize with user defined output size
    cutlass::conv::Conv2dProblemSize problem_size(      
        paras.input_size,
        paras.filter_size,
        paras.padding,
        paras.conv_stride,
        paras.dilation,
        paras.output_size(),
        mode,
        split_k_slices);

    // Construct ImplicitGemm::Argument structure with conv2d 
    // problem size, data pointers, and epilogue values
    typename ImplicitGemm::Arguments arguments{
        problem_size,
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        tensor_c.device_ref(),
        tensor_c.device_ref(),
        {paras.alpha, paras.beta},
    };


    ImplicitGemm implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CHK_CUTLASS(implicit_gemm_op.can_implement(arguments));

    CHK_CUTLASS(implicit_gemm_op.initialize(arguments, workspace.get()));

    // Launch the kernel
    CHK_CUTLASS(implicit_gemm_op());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
