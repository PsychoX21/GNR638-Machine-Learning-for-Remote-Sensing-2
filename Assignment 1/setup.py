from setuptools import setup, find_packages
import os
import sys

# Ensure build directory exists
build_dir = os.path.join(os.path.dirname(__file__), 'build')
if not os.path.exists(build_dir):
    print("ERROR: Build directory not found. Please run 'make build' first.")
    sys.exit(1)

setup(
    name='deepnet',
    version='1.0.0',
    description='Production-grade deep learning framework with C++ backend',
    author='GNR638 Student',
    python_requires='>=3.12',
    packages=find_packages(),
    package_dir={'deepnet': 'deepnet/python'},
    package_data={
        'deepnet': ['*.so', '*.pyd']  # Include compiled bindings
    },
    install_requires=[
        'opencv-python>=4.8.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
    ],
)
