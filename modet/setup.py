import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='modet',
    version='1.0',
    author='kwea123',
    author_email='kwea123@gmail.com',
    description='cppcuda example',
    long_description='cppcuda example',
    ext_modules=[
        CUDAExtension(
            name='modet',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
