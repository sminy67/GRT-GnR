# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

tnn_utils = CUDAExtension(
    name="grt_embeddings_forward",
    sources=[
        "layer/grt_embeddings_forward.cpp",
        "layer/grt_embeddings_forward_cuda.cu",
    ],
    extra_compile_args={
        "cxx": [
            "-O3",
            "-g",
            "-DUSE_MKL",
            "-m64",
            "-mfma",
            "-masm=intel",
        ],
        "nvcc": [
            "-O3",
            "-g",
            "--expt-relaxed-constexpr",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-I/usr/local/cuda-11.7/include/",
            '-gencode=arch=compute_86,code="sm_86"',
        ],
    },
)
setup(
    name="grt_embeddings_forward",
    description="grt_embeddings_forward",
    packages=find_packages(),
    ext_modules=[tnn_utils],
    cmdclass={"build_ext": BuildExtension},
)
