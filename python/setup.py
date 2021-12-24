from skbuild import setup

setup(
    name="pygwmodel",
    version="0.2.0",
    author="Yigong Hu",
    packages=["pygwmodel"],
    install_requires=['cython'],
    cmake_args=["-DWITH_PYTHON:BOOL=ON"]
)