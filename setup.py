import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

curr_path = os.path.abspath(os.path.dirname(__file__))

setup(
    name='cuda_process',
    version='0.4',
    ext_modules=[
        CUDAExtension('cuda_process', ['pywrap.cpp', 'normalize.cu', 'reduce.cu'],
                      include_dirs=[curr_path])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)