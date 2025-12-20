import os
import subprocess
import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension
import platform
import shutil

# Build the CMake dependency first
def build_cmake_dependency():
    cmake_build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmake_build")
    
    if not os.path.exists(cmake_build_dir):
        os.makedirs(cmake_build_dir)
    
    subprocess.check_call(
        ["cmake", ".."],
        cwd=cmake_build_dir
    )
    subprocess.check_call(
        ["cmake", "--build", ".", "--config", "Release"],
        cwd=cmake_build_dir
    )
    
    return {
        "include_dirs": [os.path.join(cmake_build_dir, "gloo-headers")],
        "library_dirs": [],
        "libraries": []
    }

# Build the dependency
cmake_info = build_cmake_dependency()

system = platform.system()
architecture = platform.machine()
triplet = None

if system == "Linux":
    triplet = "x64-linux"
elif system == "Darwin":
    triplet = "arm64-osx" if architecture == "arm64" else "x64-osx"
elif system == "Windows":
    triplet = "x64-windows"
else:
    raise ValueError(f"Unsupported platform: {system} on {architecture}")

vcpkg_installed = "vcpkg_installed"

sources = ["src/webgpu_backend.cpp"]

# Combine include directories from both CMake and original paths
include_dirs = [
    f"{os.path.dirname(os.path.abspath(__file__))}/include/",
    f"{os.path.dirname(os.path.abspath(__file__))}/{vcpkg_installed}/{triplet}/include"
]
include_dirs.extend(cmake_info["include_dirs"])

# Combine library directories
library_dirs = [f"{vcpkg_installed}/{triplet}/lib/"]
library_dirs.extend(cmake_info["library_dirs"])

# Combine libraries
libraries = ["fmt", "inccompute"] 
libraries.extend(cmake_info["libraries"])

# Create extension module
if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="inc_collectives",
        sources=sources,
        include_dirs=include_dirs,
        define_macros=[("USE_C10D_GLOO", None), ("IS_CUDA_BUILD", None)],
        library_dirs=library_dirs,
        libraries=libraries,
    )
else:
    module = cpp_extension.CppExtension(
        name="inc_collectives",
        sources=sources,
        define_macros=[("USE_C10D_GLOO", None)],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
    )

setup(
    name="webgpu_backend",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
