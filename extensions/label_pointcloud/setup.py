from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='label_pointcloud_extension',
    ext_modules=[
        CUDAExtension('label_pointcloud_extension', [
            'extension.cpp',
            'label_pt.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
