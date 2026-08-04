# syntax=docker/dockerfile:1

# pie CUDA runner image for RunPod — sm_90 (H100) only.
#
# Adapted from test-time-bench's backend/docker/eval-worker-base.Dockerfile and
# decoder-host.Dockerfile. What survives that template is the RunPod plumbing:
# PUBLIC_KEY-injected SSH (never a baked authorized_keys), the arch-stamped tag,
# the cache layout and the sshd entrypoint shape.
#
# What is deliberately absent is everything bound to test-time-bench's own pie
# pin: no tts-arena/main pin, no /opt/ttb-pie-rev stamp, no decoder-rust
# inferlet, no agentic-decoder wiring, no baked model metadata. Those describe a
# different pie lineage with a different linkage contract; carrying them here
# would make this image claim a provenance it does not have.
#
# Build only through scripts/build-runner-image.sh. The revision this image
# claims cannot be observed from inside the build — the context carries no git
# history — so the script is the only thing that can state it, and it is the
# script that proves the claim against the checkout before passing it in.

# One arch, spelled once, and deliberately not a list.
#
# driver/cuda/cmake/DetectCudaArchitecture.cmake normalises 90 -> 90a, because
# CUTLASS/CuTe gate every Hopper GMMA/TMA kernel on the accelerated target and a
# bare "90" silently drops them. That same file raises FATAL_ERROR when neither
# this value nor nvidia-smi is available, so a GPU-less build MUST set it.
#
# Not 80;86;89;90: driver/cuda/CMakeLists.txt turns PIE_CUDA_FLASHINFER_MAMBA_SM90
# on when ANY list entry is 90, which defines FLASHINFER_MAMBA_ENABLE_SM90 while
# __CUDA_MINIMUM_ARCH__ is 800, and the FlashInfer Mamba SSU kernels behind that
# macro do not compile at that minimum. Not 90;100 either: the same file stubs
# custom-all-reduce whenever the sm100 flag is set, which would silently drop it
# for the sm90 target we actually ship.
ARG PIE_CUDA_ARCHITECTURES=90

# CUDA 12.9 on Ubuntu 24.04 rather than the template's 12.8.1-cudnn on 22.04, for
# two reasons that both come from dev linking the toolkit dynamically:
#   - the devel image must carry NCCL's header at CONFIGURE time
#     (driver/cuda/CMakeLists.txt find_path/find_library for nccl are REQUIRED);
#     this tag ships NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.26.5-1+cuda12.9.
#   - builder and runtime must agree on toolkit sonames, so they are the same
#     CUDA version from the same publisher.
# Ubuntu 24.04 also ships cmake >= 3.23, so the template's separate CMake
# download is unnecessary here; the version is asserted below rather than assumed.
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04@sha256:632c7db01c6a392f1a4a51b575ee1c125d19e7656de83a285ac222fc35b80606 AS builder

ARG PIE_CUDA_ARCHITECTURES
# Throttle for shared build hosts: the CUDA driver is ~50 translation units and
# parallel nvcc is this build's memory high-water mark. Empty means "use every
# core". Applied as CARGO_BUILD_JOBS at the build step, because cargo derives
# NUM_JOBS from it and the cmake crate takes its parallelism from NUM_JOBS — so
# one value throttles both halves.
ARG BUILD_JOBS=

ENV CUDA_HOME=/usr/local/cuda \
    CMAKE_CUDA_ARCHITECTURES=${PIE_CUDA_ARCHITECTURES} \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH} \
    DEBIAN_FRONTEND=noninteractive

# No libssl-dev: every reqwest in this workspace is built with rustls and with
# default-features off precisely so nothing links openssl (runtime/engine's
# manifest says so in as many words). Adding it would invite an openssl-sys
# build that the tree deliberately does not want.
#
# No libnccl* either. The CUDA devel base already pins a matching NCCL, and the
# NVIDIA apt repo will happily upgrade the whole CUDA branch out from under the
# toolkit we link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        ninja-build \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*
# ninja is installed but the generator is deliberately NOT forced: the cmake
# crate picks Unix Makefiles by default, which is what pie's own developers and
# CI build with. Switching generators is a behaviour change to make on evidence,
# not in passing.

# Fail in seconds, not thirty minutes in. These are configure-time requirements of
# driver/cuda/CMakeLists.txt, and a missing one otherwise surfaces only after the
# toolchain and the dependency fetch have already run.
#
# PIE_CUDA_ARCHITECTURES is checked HERE and not only in the wrapper script,
# because it is the argument whose misuse is silent. A direct
# `docker build --build-arg PIE_CUDA_ARCHITECTURES=90;100` produces an image that
# ships sm90 with custom-all-reduce stubbed out (CMakeLists.txt:364-368 disables it
# whenever the sm100 flag is set) and nothing about the resulting image says so;
# 80;86;89;90 instead breaks the compile outright, via the any-arch SM90 gate. One
# numeric value only — CMake appends the accelerated `a` suffix itself, so a
# caller never spells it.
RUN set -eu; \
    printf '%s' "${PIE_CUDA_ARCHITECTURES}" | grep -Eq '^[0-9]{2,3}$' || { \
        echo "ERROR: PIE_CUDA_ARCHITECTURES must be exactly one numeric architecture, got '${PIE_CUDA_ARCHITECTURES}'." >&2; \
        echo "  A list is never correct here: 80;86;89;90 defines FLASHINFER_MAMBA_ENABLE_SM90 while" >&2; \
        echo "  __CUDA_MINIMUM_ARCH__ is 800 and fails to compile, and 90;100 silently stubs" >&2; \
        echo "  custom-all-reduce for the sm90 target the image ships." >&2; \
        exit 1; }; \
    cmake --version | head -n1; \
    cmake_version="$(cmake --version | head -n1 | awk '{print $3}')"; \
    [ "$(printf '3.23\n%s\n' "${cmake_version}" | sort -V | head -n1)" = "3.23" ] \
        || { echo "ERROR: cmake ${cmake_version} < 3.23 required by driver/cuda/CMakeLists.txt" >&2; exit 1; }; \
    for header in /usr/include/nccl.h "${CUDA_HOME}/include/nccl.h"; do \
        if [ -r "${header}" ]; then found_header=1; break; fi; \
    done; \
    [ -n "${found_header:-}" ] \
        || { echo "ERROR: nccl.h not found; driver/cuda/CMakeLists.txt find_path(NCCL_INCLUDE_DIR) is REQUIRED" >&2; exit 1; }; \
    ldconfig -p | grep -q 'libnccl\.so' \
        || { echo "ERROR: libnccl.so not found; driver/cuda/CMakeLists.txt find_library(NCCL_LIBRARY) is REQUIRED" >&2; exit 1; }

# The toolchain is a property of the source tree, not of this Dockerfile:
# rust-toolchain.toml pins channel 1.97.1 with rustfmt/clippy and the
# wasm32-wasip2 target. Installing with --default-toolchain none and then asking
# rustup to resolve inside the repo is what makes the file authoritative; a bare
# `stable` here would silently build with a different compiler than CI uses.
WORKDIR /src
COPY rust-toolchain.toml ./
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain none \
    && cargo --version
# Then assert the file was actually honoured. Whether a given rustup release
# auto-installs on `rustup show` has changed over time, and a toolchain that
# quietly resolved to something else would compile this image with a different
# compiler than CI uses. The channel is read back out of the file rather than
# restated here, so these lines cannot drift from it.
RUN set -eu; \
    channel="$(sed -n 's/^channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' rust-toolchain.toml)"; \
    [ -n "${channel}" ] || { echo "ERROR: rust-toolchain.toml names no channel" >&2; exit 1; }; \
    rustc --version; \
    rustc --version | grep -q -- "${channel}" \
        || { echo "ERROR: active toolchain is not ${channel} from rust-toolchain.toml" >&2; exit 1; }; \
    rustup target list --installed | grep -qx 'wasm32-wasip2' \
        || { echo "ERROR: wasm32-wasip2 missing; rust-toolchain.toml declares it" >&2; exit 1; }; \
    for component in rustfmt clippy; do \
        rustup component list --installed | grep -q "^${component}" \
            || { echo "ERROR: ${component} missing; rust-toolchain.toml declares it" >&2; exit 1; }; \
    done

COPY . .

# The rev is an input, not an observation: the build context carries no git
# history, so nothing in here can derive it. Reject anything that is not a
# 40-hex commit so a mistyped or empty --build-arg cannot become a stamped lie.
ARG PIE_REV
RUN printf '%s' "${PIE_REV}" | grep -Eq '^[0-9a-f]{40}$' \
    || { echo "ERROR: PIE_REV must be a 40-character lowercase commit sha (got '${PIE_REV}')" >&2; exit 1; }

# driver-cuda is not a default feature on dev (bin/pie/Cargo.toml: default = []),
# so a plain build produces a binary whose only flavor is dummy.
# --no-default-features keeps that explicit rather than incidental.
#
# No --locked: this workspace tracks no root Cargo.lock, so the dependency
# closure resolves at build time. Pinning PIE_REV pins pie's source and not its
# dependencies; any comparability claim made about this image has to say so.
#
# ccache is found by driver/cuda/CMakeLists.txt on its own; the cache mount is
# what makes it worth anything across builds. sccache is deliberately not used —
# that file rejects it for nvcc.
# The environment is set with `export`, not with assignment words in front of the
# command. A shell recognizes assignment words BEFORE parameter expansion, so a
# `${BUILD_JOBS:+CARGO_BUILD_JOBS=${BUILD_JOBS}}` prefix does not become an
# assignment when it expands — it becomes the command name, and the step dies
# with exit 127. Do not reintroduce that shape.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/CPM \
    set -eu; \
    export CCACHE_DIR=/root/.cache/ccache; \
    export CPM_SOURCE_CACHE=/root/.cache/CPM; \
    if [ -n "${BUILD_JOBS:-}" ]; then export CARGO_BUILD_JOBS="${BUILD_JOBS}"; fi; \
    cargo install --path bin/pie --root /usr/local \
        --no-default-features --features driver-cuda

# Pre-provision the embedded Python-WASM runtime, and bake the default config,
# so a fresh pod neither downloads a runtime nor starts unconfigured.
#
# Both have to happen in the builder. Running the binary at all needs
# libcuda.so.1, which the NVIDIA container toolkit supplies at `docker run` and
# no image carries; only the devel stage has a stub to stand in for it at build
# time. Both land under PIE_HOME, which the runtime stage copies wholesale.
#
# `pie config init` renders its template against the compiled flavors, so with
# driver-cuda linked the generated [worker.model.driver] block is the CUDA one.
# The model it names is the template's placeholder: weights are deliberately not
# baked, and whoever runs this supplies the real artifact.
ENV PIE_HOME=/root/.cache/pie
RUN set -eu; \
    ln -sf "${CUDA_HOME}/lib64/stubs/libcuda.so" "${CUDA_HOME}/lib64/stubs/libcuda.so.1"; \
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH:-}"; \
    /usr/local/bin/pie runtime install; \
    /usr/local/bin/pie config init --force; \
    /usr/local/bin/pie config show


FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04@sha256:3540120e2c1c7c234194487bbcd491bf22da21215390fd4e497b95502c9e2277 AS runtime

ARG PIE_CUDA_ARCHITECTURES
ARG PIE_REV
# Where the build tooling itself came from. Once this Dockerfile is committed on
# top of the pinned pie revision, HEAD is no longer that revision, and the two
# facts stop being interchangeable: the binary is still built from PIE_REV's
# source, but the recipe is not. Recording only one of them would make the image's
# provenance ambiguous exactly when it starts to matter.
#
# Two shapes, distinguished by prefix, because pre-commit there is no commit to
# name at all:
#   <40-hex>      a commit that contains the tooling
#   tree:<40-hex> a git tree holding the exact tooling that ran, used when the
#                 tooling is not yet committed — recoverable with
#                 `git ls-tree -r <tree>`, and refused by --push
#
# This does NOT default to PIE_REV. Doing so was a provenance bug: in the
# pre-commit case, which is precisely the case a default is for, PIE_REV's tree
# contains no docker/ directory and no build script, so the label named a commit
# that could not produce the image. scripts/build-runner-image.sh is what knows
# which shape is true, so it must pass the value in.
ARG PIE_TOOLING_REV

LABEL org.opencontainers.image.source="https://github.com/pie-project/pie" \
      org.opencontainers.image.revision="${PIE_REV}" \
      org.pie-project.image.tooling_revision="${PIE_TOOLING_REV}" \
      org.pie-project.cuda.architectures="${PIE_CUDA_ARCHITECTURES}" \
      org.pie-project.pie.features="driver-cuda"

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PIE_BIN=/usr/local/bin/pie \
    PIE_HOME=/root/.cache/pie

# No libssl here either: the binary is rustls-only. libgomp1 is the host-side
# OpenMP runtime the CUDA driver's C++ may pull; the linker assertion at the end
# of this stage is what actually decides whether this list is complete.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /root/.cache/pie /root/.cache/huggingface /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -ri 's/^#?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -ri 's/^#?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && sed -ri 's/^#?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && ssh-keygen -A

COPY --from=builder /usr/local/bin/pie /usr/local/bin/pie
# Carries both the pre-provisioned python-WASM runtime and the baked config.
COPY --from=builder /root/.cache/pie /root/.cache/pie

# Nothing is copied forward from the CUDA toolkit. Every library dev links
# dynamically is already in this base, checked against the image rather than
# assumed: cuda-cudart-12-9, libcublas-12-9 (which provides libcublasLt),
# cuda-nvrtc-12-9 and libnccl2 are all installed, and libnvrtc.so.12 with its
# builtins is present on disk. The assertion below is what keeps that true.

COPY docker/runpod-sshd-entrypoint.sh /usr/local/bin/runpod-sshd-entrypoint
RUN printf '#!/bin/sh\nset -eu\nexec sleep infinity\n' > /usr/local/bin/keepalive \
    && chmod +x /usr/local/bin/pie /usr/local/bin/keepalive /usr/local/bin/runpod-sshd-entrypoint

# The runtime base is not required by anyone to carry what dev dynamically links
# — the tts-arena/main lineage static-linked the toolkit, so no earlier image
# ever had to. Rather than trusting a package list, ask the linker: if any
# soname besides libcuda.so.1 is unresolved, this image cannot run pie and the
# build fails here instead of on a rented GPU. libcuda.so.1 is the one legitimate
# absence — it comes from the host's NVIDIA driver via the container toolkit,
# never from the image.
# The check is "everything this binary needs resolves", not "these five names are
# present". Which toolkit libraries end up as NEEDED is the linker's decision —
# --as-needed drops any whose symbols went unreferenced — so demanding a fixed
# list would invent a failure for a library the binary legitimately does not
# load. The unresolved set is the fact that matters; the table is kept in the
# image so the same claim can be re-checked on the pod.
RUN set -eu; \
    ldd /usr/local/bin/pie > /opt/pie-linkage.txt; \
    unresolved="$(awk '/not found/ { print $1 }' /opt/pie-linkage.txt | grep -v '^libcuda\.so\.1$' || true)"; \
    if [ -n "${unresolved}" ]; then \
        echo "ERROR: unresolved shared libraries in the runtime stage:" >&2; \
        printf '  %s\n' ${unresolved} >&2; \
        echo "dev dynamic-links the CUDA toolkit (worker/build.rs), so this base must carry them." >&2; \
        exit 1; \
    fi; \
    echo "CUDA toolkit libraries this binary needs, and where they resolved:"; \
    grep -E '(libcudart|libcublas|libcublasLt|libnvrtc|libnccl|libcuda)\.so' /opt/pie-linkage.txt || true; \
    grep -q 'libcuda\.so\.1 => not found' /opt/pie-linkage.txt \
        || echo "note: libcuda.so.1 resolves inside the image; normally the container toolkit injects it at run time"

# Our own provenance, replacing the template's /opt/ttb-pie-rev: the exact pie
# commit this binary was built from, plus where the recipe came from. Written from
# the same ARGs the labels carry, so the files and the labels cannot disagree.
#
# PIE_TOOLING_REV has no default on purpose, so it is checked rather than assumed:
# an unset value would label the image with an empty provenance string, which is a
# quieter version of the same lie as a wrong one.
RUN set -eu; \
    printf '%s' "${PIE_TOOLING_REV}" | grep -Eq '^(tree:)?[0-9a-f]{40}$' \
        || { echo "ERROR: PIE_TOOLING_REV must be a 40-character lowercase sha, optionally 'tree:'-prefixed (got '${PIE_TOOLING_REV}')." >&2; \
             echo "  Build through scripts/build-runner-image.sh; it is what knows whether the tooling is committed." >&2; \
             exit 1; }; \
    printf '%s\n' "${PIE_REV}" > /opt/pie-rev; \
    printf '%s\n' "${PIE_TOOLING_REV}" > /opt/pie-tooling-rev

EXPOSE 22/tcp 8080/tcp
ENTRYPOINT ["/usr/local/bin/runpod-sshd-entrypoint"]
CMD ["-D", "-e"]
