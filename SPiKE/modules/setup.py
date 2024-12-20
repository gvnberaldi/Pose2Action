# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

''' Modified based on: https://github.com/hehefan/P4Transformer/ '''
# Get the directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))

_ext_src_root = os.path.join(script_dir, '_ext_src')
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))
print(f"ext root: {_ext_src_root}")
print(f"ext source: {_ext_sources}")
print(f"ext headers: {_ext_headers}")

headers = "-I" + os.path.join(script_dir, '_ext_src', 'include')
print ("!!!!!!", headers)

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", headers],
                "nvcc": ["-O2", headers],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    # Specify build_temp and build_lib to customize the build directories
    script_args=["build_ext", "--build-temp", os.path.join(script_dir, "build"), "--build-lib", os.path.join(script_dir, "build_lib")]
)
