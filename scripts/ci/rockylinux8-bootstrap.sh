#!/usr/bin/env bash
# The manylinux_2_28 build host: a `rockylinux:8`-based container (glibc
# 2.28) with gcc-toolset-12, cmake/ninja, gh and a Rust toolchain. Every
# Linux binary pie publishes is built here so it runs on RHEL 8 / Ubuntu
# 20.04 / Debian 11 and newer. Sourced by the workflows; appends to
# GITHUB_ENV and GITHUB_PATH.
set -euxo pipefail

dnf install -y epel-release dnf-plugins-core
dnf config-manager --set-enabled powertools
dnf install -y --setopt=install_weak_deps=False \
  openssl-devel pkg-config perl git patch cmake ninja-build \
  gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ \
  python3.12 python3.12-pip
alternatives --set python3 /usr/bin/python3.12
# `gh` uploads release assets the same way the desktop legs do.
dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
dnf install -y --setopt=install_weak_deps=False gh

# Pin the compiler explicitly: cmake-rs and nvcc otherwise pick the stock
# gcc-8.5 (no C++20), and rustc's final link must go through the toolset's
# gcc driver so libstdc++_nonshared.a is pulled in for the C++17/20 symbols.
triple="${1:?target triple}"
{
  echo "CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc"
  echo "CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++"
  echo "AR=/opt/rh/gcc-toolset-12/root/usr/bin/ar"
  echo "CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++"
  echo "PATH=/opt/rh/gcc-toolset-12/root/usr/bin:${PATH}"
  echo "LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:/opt/rh/gcc-toolset-12/root/usr/lib"
  echo "PKG_CONFIG_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64/pkgconfig"
  echo "CARGO_TARGET_$(echo "$triple" | tr 'a-z-' 'A-Z_')_LINKER=/opt/rh/gcc-toolset-12/root/usr/bin/gcc"
  echo "CARGO_INCREMENTAL=0"
} >> "$GITHUB_ENV"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
