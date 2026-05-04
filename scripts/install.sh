#!/usr/bin/env bash
# pie installer — served at https://pie-project.org/install.sh
#
#   curl -fsSL https://pie-project.org/install.sh | bash
#   curl -fsSL https://pie-project.org/install.sh | PIE_FLAVOR=cuda12.8 bash
#   curl -fsSL https://pie-project.org/install.sh | PIE_VERSION=v0.1.2 bash
#
# Environment overrides:
#   PIE_VERSION       Release tag (default: latest-build).
#   PIE_FLAVOR        portable | cuda{12.6,12.8,13.0}
#                     | portable-cuda{12.6,12.8,13.0}. Auto-detected when unset.
#   PIE_INSTALL_DIR   Install location for the `pie` binary (default: ~/.local/bin).
#   PIE_REPO          GitHub owner/name (default: pie-project/pie).
#   PIE_DOWNLOAD_BASE Override the asset base URL (default: GitHub releases).

set -euo pipefail

PIE_REPO="${PIE_REPO:-pie-project/pie}"
PIE_VERSION="${PIE_VERSION:-latest-build}"
PIE_INSTALL_DIR="${PIE_INSTALL_DIR:-${HOME}/.local/bin}"
PIE_DOWNLOAD_BASE="${PIE_DOWNLOAD_BASE:-https://github.com/${PIE_REPO}/releases/download/${PIE_VERSION}}"

bold=""; reset=""
if [ -t 2 ]; then bold=$'\033[1m'; reset=$'\033[0m'; fi
info() { printf '%s==>%s %s\n' "$bold" "$reset" "$*" >&2; }
err()  { printf '%serror:%s %s\n' "$bold" "$reset" "$*" >&2; exit 1; }

if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -q -O "$2" "$1"; }
else
  err "neither curl nor wget found; install one and re-run"
fi

for tool in uname tar mktemp install; do
  command -v "$tool" >/dev/null 2>&1 || err "missing required tool: $tool"
done

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

os_raw="$(uname -s)"
arch_raw="$(uname -m)"

case "$os_raw" in
  Linux)               os=linux ;;
  Darwin)              os=darwin ;;
  MINGW*|MSYS*|CYGWIN*)
    err "Windows is not supported. Use WSL and the Linux build." ;;
  *)
    err "unsupported operating system: $os_raw" ;;
esac

case "$arch_raw" in
  x86_64|amd64)        arch=x86_64 ;;
  aarch64|arm64)       arch=aarch64 ;;
  *)
    err "unsupported architecture: $arch_raw" ;;
esac

if [ "$os" = darwin ] && [ "$arch" != aarch64 ]; then
  err "Intel Macs are not supported; Apple Silicon (arm64) only."
fi

# ---------------------------------------------------------------------------
# Auto-pick flavor (Linux x86_64 only)
#
# CUDA toolkit -> minimum driver (Linux):
#   13.0 -> 580.65.06
#   12.x -> 525.60.13
# Pick the highest toolkit the installed driver supports; otherwise fall
# back to the portable build. Users override with PIE_FLAVOR.
# ---------------------------------------------------------------------------

detect_cuda_flavor() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local driver major
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  [ -n "$driver" ] || return 0
  major="${driver%%.*}"
  case "$major" in *[!0-9]*|"") return 0 ;; esac
  if [ "$major" -ge 580 ]; then
    echo cuda13.0
  elif [ "$major" -ge 525 ]; then
    echo cuda12.8
  else
    info "NVIDIA driver $driver is older than 525 (CUDA 12 minimum); using portable build"
  fi
}

if [ -z "${PIE_FLAVOR:-}" ]; then
  if [ "$os" = linux ] && [ "$arch" = x86_64 ]; then
    PIE_FLAVOR="$(detect_cuda_flavor || true)"
    PIE_FLAVOR="${PIE_FLAVOR:-portable}"
  else
    PIE_FLAVOR=portable
  fi
fi

# Map (os, arch, flavor) -> release asset name. Names mirror what
# .github/workflows/build.yml uploads.
asset_for() {
  case "$os/$arch/$1" in
    linux/x86_64/portable)            echo "pie-x86_64-manylinux_2_28.tar.gz" ;;
    linux/aarch64/portable)           echo "pie-aarch64-manylinux_2_28.tar.gz" ;;
    darwin/aarch64/portable)          echo "pie-aarch64-darwin.tar.gz" ;;
    linux/x86_64/cuda12.6 \
    | linux/x86_64/cuda12.8 \
    | linux/x86_64/cuda13.0 \
    | linux/x86_64/portable-cuda12.6 \
    | linux/x86_64/portable-cuda12.8 \
    | linux/x86_64/portable-cuda13.0)
      echo "pie-x86_64-linux-${1}.tar.gz" ;;
    *) return 1 ;;
  esac
}

asset="$(asset_for "$PIE_FLAVOR")" || err \
  "no '$PIE_FLAVOR' build for $os/$arch. Valid flavors: portable; or (Linux x86_64 only) cuda12.6, cuda12.8, cuda13.0, portable-cuda12.6, portable-cuda12.8, portable-cuda13.0."

url="${PIE_DOWNLOAD_BASE}/${asset}"
info "platform: ${os}/${arch}"
info "flavor:   ${PIE_FLAVOR}"
info "version:  ${PIE_VERSION}"
info "asset:    ${url}"

# ---------------------------------------------------------------------------
# Download + install
# ---------------------------------------------------------------------------

tmp="$(mktemp -d -t pie-install.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT

info "downloading…"
fetch "$url" "${tmp}/${asset}"

info "verifying archive…"
tar -tzf "${tmp}/${asset}" >/dev/null

tar -C "${tmp}" -xzf "${tmp}/${asset}"
[ -f "${tmp}/pie" ] || err "archive did not contain a 'pie' binary"

mkdir -p "${PIE_INSTALL_DIR}"
install -m 0755 "${tmp}/pie" "${PIE_INSTALL_DIR}/pie"

info "installed: ${PIE_INSTALL_DIR}/pie"

case ":${PATH}:" in
  *":${PIE_INSTALL_DIR}:"*) ;;
  *)
    info "${PIE_INSTALL_DIR} is not on your PATH. Add it with e.g.:"
    info "  echo 'export PATH=\"${PIE_INSTALL_DIR}:\$PATH\"' >> ~/.bashrc"
    ;;
esac
