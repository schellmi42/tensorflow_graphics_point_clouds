licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "pointcloud_common_kernels",
    srcs = [
        "cc/kernels/cuda_kernel_utils.h",
        "cc/kernels/defines.hpp",
        "cc/kernels/gpu_device.hpp",
        "cc/kernels/tf_utils.hpp",
        "cc/kernels/tf_gpu_device.hpp",
        "cc/kernels/math_helper.h",
        "cc/kernels/grid_utils.h",
        "cc/kernels/rnd_utils.h",
        "cc/kernels/nn_utils.h",
    ],
    # visibility = ["//tfg_custom_ops/compute_keys:__pkg__"],
)

