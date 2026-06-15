#!/usr/bin/env bash
# pie installer — served at https://pie-project.org/install.sh
#
#   curl -fsSL https://pie-project.org/install.sh | bash
#   curl -fsSL https://pie-project.org/install.sh | PIE_FLAVOR=cuda12.8 bash
#   curl -fsSL https://pie-project.org/install.sh | PIE_VERSION=0.4.0 bash
#
# Environment overrides:
#   PIE_VERSION       Release tag (default: 0.4.0).
#   PIE_FLAVOR        portable | cuda{12.8,13.0}. Auto-detected when unset.
#   PIE_CC            GPU compute capability for CUDA flavors, e.g. 90, 100
#                     (auto-detected via nvidia-smi; selects the per-CC binary).
#   PIE_INSTALL_DIR   Install location for the `pie` binary (default: ~/.local/bin).
#   PIE_REPO          GitHub owner/name (default: pie-project/pie).
#   PIE_DOWNLOAD_BASE Override the asset base URL (default: GitHub releases).

set -euo pipefail

PIE_REPO="${PIE_REPO:-pie-project/pie}"
PIE_VERSION="${PIE_VERSION:-0.4.0}"
PIE_INSTALL_DIR="${PIE_INSTALL_DIR:-${HOME}/.local/bin}"
PIE_DOWNLOAD_BASE="${PIE_DOWNLOAD_BASE:-https://github.com/${PIE_REPO}/releases/download/${PIE_VERSION}}"
PIE_DETECTED_FLAVOR=""
PIE_FLAVOR_REASON=""

bold=""; dim=""; red=""; green=""; yellow=""; reset=""
if [ -t 2 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-}" != "dumb" ]; then
  bold=$'\033[1m'
  dim=$'\033[2m'
  red=$'\033[31m'
  green=$'\033[32m'
  yellow=$'\033[33m'
  reset=$'\033[0m'
fi

heading() { printf '\n%s%s%s\n' "$bold" "$*" "$reset" >&2; }
detail()  { printf '  %s%s%s\n' "$dim" "$*" "$reset" >&2; }
step()    { printf '%s==>%s %s\n' "$bold" "$reset" "$*" >&2; }
ok()      { printf '%ssuccess:%s %s\n' "$green" "$reset" "$*" >&2; }
warn()    { printf '%swarning:%s %s\n' "$yellow" "$reset" "$*" >&2; }
err()     { printf '%serror:%s %s\n' "$red" "$reset" "$*" >&2; exit 1; }
debug()   { if [ -n "${PIE_VERBOSE:-}" ]; then detail "$*"; fi; }

if command -v curl >/dev/null 2>&1; then
  if [ -t 2 ]; then
    fetch() { curl -fL --progress-bar --retry 3 -o "$2" "$1"; }
  else
    fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
  fi
elif command -v wget >/dev/null 2>&1; then
  if [ -t 2 ]; then
    fetch() { wget -q --show-progress -O "$2" "$1"; }
  else
    fetch() { wget -q -O "$2" "$1"; }
  fi
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
  command -v nvidia-smi >/dev/null 2>&1 || {
    PIE_FLAVOR_REASON="no NVIDIA driver detected"
    return 0
  }
  local driver major
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  [ -n "$driver" ] || {
    PIE_FLAVOR_REASON="could not read NVIDIA driver version"
    return 0
  }
  major="${driver%%.*}"
  case "$major" in
    *[!0-9]*|"")
      PIE_FLAVOR_REASON="could not parse NVIDIA driver version '$driver'"
      return 0
      ;;
  esac
  if [ "$major" -ge 580 ]; then
    PIE_FLAVOR_REASON="NVIDIA driver $driver supports CUDA 13"
    PIE_DETECTED_FLAVOR=cuda13.0
  elif [ "$major" -ge 525 ]; then
    PIE_FLAVOR_REASON="NVIDIA driver $driver supports CUDA 12"
    PIE_DETECTED_FLAVOR=cuda12.8
  else
    PIE_FLAVOR_REASON="NVIDIA driver $driver is older than 525 (CUDA 12 minimum)"
  fi
}

# Compute capability of the first GPU, normalized to the SM string used in the
# per-CC asset names ("9.0" -> "90", "12.0" -> "120"). Override with PIE_CC.
detect_cuda_cc() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .' || true)"
  case "$cap" in
    *[!0-9]*|"") return 0 ;;
  esac
  PIE_DETECTED_CC="$cap"
}

if [ -z "${PIE_FLAVOR:-}" ]; then
  if [ "$os" = linux ] && { [ "$arch" = x86_64 ] || [ "$arch" = aarch64 ]; }; then
    detect_cuda_flavor || true
    PIE_FLAVOR="${PIE_DETECTED_FLAVOR:-portable}"
  else
    PIE_FLAVOR=portable
  fi
fi

# Detect the GPU compute capability so CUDA flavors can pick the per-CC binary.
if [ -z "${PIE_CC:-}" ]; then
  detect_cuda_cc || true
  PIE_CC="${PIE_DETECTED_CC:-}"
fi

# Map (os, arch, flavor) -> one or more candidate release asset names, in
# preference order (mirrors what .github/workflows/build.yml uploads). CUDA
# flavors prefer the GH-hosted manylinux_2_28 build for the detected compute
# capability (broad glibc compat, small), then fall back to the all-arch
# native linux-x64 fatbin from the self-hosted GPU runner.
assets_for() {
  case "$os/$arch/$1" in
    linux/x86_64/portable)            echo "pie-x86_64-linux-vulkan.tar.gz" ;;
    linux/aarch64/portable)           echo "pie-aarch64-linux-vulkan.tar.gz" ;;
    darwin/aarch64/portable)          echo "pie-aarch64-darwin.tar.gz" ;;
    linux/x86_64/cuda12.8 | linux/x86_64/cuda13.0 \
    | linux/aarch64/cuda12.8 | linux/aarch64/cuda13.0)
      # Per-compute-capability binary first (small, arch-matched) when the CC
      # was detected; then the all-arch fatbin fallback (self-hosted, x86_64).
      if [ -n "${PIE_CC:-}" ]; then
        echo "pie-${arch}-manylinux_2_28-${1}-sm${PIE_CC}.tar.gz"
      fi
      if [ "$arch" = x86_64 ]; then
        echo "pie-x86_64-linux-${1}.tar.gz"
      fi
      ;;
    *) return 1 ;;
  esac
}

candidates="$(assets_for "$PIE_FLAVOR")" || err \
  "no '$PIE_FLAVOR' build for $os/$arch. Valid flavors: portable; or (Linux) cuda12.8, cuda13.0."

heading "Installing Pie"
detail "Version:  ${PIE_VERSION}"
detail "Platform: ${os}/${arch}"
detail "Flavor:   ${PIE_FLAVOR}"
[ -n "$PIE_FLAVOR_REASON" ] && detail "Reason:   ${PIE_FLAVOR_REASON}"
detail "Target:   ${PIE_INSTALL_DIR}/pie"

# ---------------------------------------------------------------------------
# Download + install
# ---------------------------------------------------------------------------

tmp="$(mktemp -d -t pie-install.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT

step "Downloading release archive"
asset=""
url=""
for cand in $candidates; do
  cand_url="${PIE_DOWNLOAD_BASE}/${cand}"
  debug "Trying:   ${cand_url}"
  if fetch "$cand_url" "${tmp}/${cand}"; then
    asset="$cand"; url="$cand_url"; break
  fi
done
if [ -z "$asset" ]; then
  err "download failed for flavor '${PIE_FLAVOR}' (tried: ${candidates}) under ${PIE_DOWNLOAD_BASE}. Check PIE_VERSION=${PIE_VERSION} and PIE_FLAVOR=${PIE_FLAVOR}, then try again."
fi
debug "Asset:    ${url}"

step "Verifying archive"
if ! tar -tzf "${tmp}/${asset}" >/dev/null; then
  err "downloaded file is not a valid gzip tar archive: ${url}"
fi

tar -C "${tmp}" -xzf "${tmp}/${asset}"
[ -f "${tmp}/pie" ] || err "archive did not contain a 'pie' binary"

step "Installing binary"
if ! mkdir -p "${PIE_INSTALL_DIR}"; then
  err "could not create install directory: ${PIE_INSTALL_DIR}"
fi
if ! install -m 0755 "${tmp}/pie" "${PIE_INSTALL_DIR}/pie"; then
  err "could not install pie to ${PIE_INSTALL_DIR}/pie. Try a user-writable PIE_INSTALL_DIR, for example: PIE_INSTALL_DIR=\"\$HOME/.local/bin\""
fi

ok "Pie was installed to ${PIE_INSTALL_DIR}/pie"

path_for_shell() {
  case "${PIE_INSTALL_DIR}" in
    "${HOME}") printf '$HOME' ;;
    "${HOME}"/*) printf '$HOME/%s' "${PIE_INSTALL_DIR#"${HOME}/"}" ;;
    *) printf '%s' "${PIE_INSTALL_DIR}" ;;
  esac
}

profile_hint() {
  local shell_name
  shell_name="$(basename "${SHELL:-}")"
  case "$shell_name" in
    zsh)  printf '%s' "$HOME/.zshrc" ;;
    bash)
      if [ "$os" = darwin ]; then
        printf '%s' "$HOME/.bash_profile"
      else
        printf '%s' "$HOME/.bashrc"
      fi
      ;;
    fish) printf '%s' "$HOME/.config/fish/config.fish" ;;
    *)    printf '%s' "$HOME/.profile" ;;
  esac
}

case ":${PATH}:" in
  *":${PIE_INSTALL_DIR}:"*) ;;
  *)
    shell_name="$(basename "${SHELL:-}")"
    install_dir_expr="$(path_for_shell)"
    profile="$(profile_hint)"
    warn "${PIE_INSTALL_DIR} is not on your PATH yet"
    printf '\nTo use pie in this terminal, run:\n' >&2
    if [ "$shell_name" = fish ]; then
      printf '  set -gx PATH "%s" $PATH\n' "$install_dir_expr" >&2
      printf '\nTo make that permanent, run:\n' >&2
      printf '  mkdir -p "%s"\n' "${profile%/*}" >&2
      printf '  echo '\''fish_add_path "%s"'\'' >> "%s"\n' "$install_dir_expr" "$profile" >&2
    else
      printf '  export PATH="%s:$PATH"\n' "$install_dir_expr" >&2
      printf '\nTo make that permanent, run:\n' >&2
      printf '  echo '\''export PATH="%s:$PATH"'\'' >> "%s"\n' "$install_dir_expr" "$profile" >&2
    fi
    ;;
esac

heading "Next steps"
detail "pie --version"
detail "pie config init"
detail "pie doctor"
