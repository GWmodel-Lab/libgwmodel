# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.241.1/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11, Ubuntu 18.04/22.04 on local arm64/Apple Silicon): debian-11, debian-10, ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

# [Optional] Install CMake version different from what base image has already installed. 
# CMake reinstall choices: none, 3.21.5, 3.22.2, or versions from https://cmake.org/download/
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/
RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libopenblas-dev libgsl-dev

RUN mkdir armadillo && \
    wget -qO- http://sourceforge.net/projects/arma/files/armadillo-9.900.6.tar.xz | \
    tar -Jxf- --directory=armadillo --strip-components=1 && \
    cmake -B armadillo/build -S armadillo -DCMAKE_INSTALL_PREFIX=/usr -DDETECT_HDF5=OFF -DBUILD_SHARED_LIBS=OFF . && \
    cmake --build armadillo/build --config Release && \
    cmake --install armadillo/build

RUN mkdir catch2 && \
    wget -qO- https://github.com/catchorg/Catch2/archive/refs/tags/v2.13.9.tar.gz | \
    tar -zxf- --directory=catch2 --strip-components=1 && \
    cmake -B catch2/build -S catch2 -DCMAKE_INSTALL_PREFIX=/usr . && \
    cmake --build catch2/build --config Release && \
    cmake --install catch2/build
