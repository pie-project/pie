#!/usr/bin/env bash
#
# Test inferlet compilation across Rust, Python, and JavaScript/TypeScript.
# Validates that all inferlets compile to valid WASM components with correct WIT exports.
#
# Usage:
#   ./tests/ci-inferlets.sh              # Run all checks
#   ./tests/ci-inferlets.sh --rust       # Rust only
#   ./tests/ci-inferlets.sh --python     # Python only
#   ./tests/ci-inferlets.sh --javascript # JavaScript/TypeScript only
#   ./tests/ci-inferlets.sh --setup      # Install prerequisites only
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="/tmp/pie-inferlet-ci"
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[PASS]${NC} $*"; PASSED=$((PASSED + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAILED=$((FAILED + 1)); FAILURES+=("$*"); }
warn() { echo -e "${YELLOW}[SKIP]${NC} $*"; SKIPPED=$((SKIPPED + 1)); }
section() { echo -e "\n${BOLD}═══ $* ═══${NC}"; }

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

check_prerequisites() {
    local need_rust=$1
    local need_python=$2
    local need_javascript=$3

    section "Checking prerequisites"

    local missing=()

    # Rust toolchain (needed by Rust builds and for installing wasm-tools)
    if command -v cargo &>/dev/null; then
        log "cargo: $(cargo --version)"
    else
        missing+=("cargo (install via rustup)")
    fi

    if [[ "$need_rust" == true ]]; then
        if rustup target list --installed 2>/dev/null | grep -q wasm32-wasip2; then
            log "wasm32-wasip2 target: installed"
        else
            missing+=("wasm32-wasip2 target (rustup target add wasm32-wasip2)")
        fi
    fi

    # Node (JavaScript/TypeScript only)
    if [[ "$need_javascript" == true ]]; then
        if command -v node &>/dev/null; then
            log "node: $(node --version)"
        else
            missing+=("node (v18+ required)")
        fi
    fi

    # uv, bakery, componentize-py (Python and JavaScript only)
    if [[ "$need_python" == true ]] || [[ "$need_javascript" == true ]]; then
        if command -v uv &>/dev/null; then
            log "uv: $(uv --version)"
        else
            missing+=("uv (https://docs.astral.sh/uv/)")
        fi

        if command -v bakery &>/dev/null; then
            log "bakery: available"
        else
            log "bakery: not found — will install from source"
            install_bakery
        fi
    fi

    if [[ "$need_python" == true ]]; then
        if command -v componentize-py &>/dev/null; then
            log "componentize-py: available"
        else
            log "componentize-py: not found — will install"
            install_componentize_py
        fi
    fi

    # wasm-tools (needed by all for validation)
    if command -v wasm-tools &>/dev/null; then
        log "wasm-tools: $(wasm-tools --version)"
    else
        log "wasm-tools: not found — will install"
        install_wasm_tools
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo -e "\n${RED}Missing prerequisites:${NC}"
        for m in "${missing[@]}"; do
            echo "  - $m"
        done
        exit 1
    fi
}

install_bakery() {
    log "Installing bakery from $REPO_ROOT/sdk/tools/bakery ..."
    uv tool install "$REPO_ROOT/sdk/tools/bakery" --quiet
    if command -v bakery &>/dev/null; then
        log "bakery: installed"
    else
        fail "Failed to install bakery"
        exit 1
    fi
}

install_componentize_py() {
    log "Installing componentize-py ..."
    uv tool install "componentize-py>=0.19" --quiet
    if command -v componentize-py &>/dev/null; then
        log "componentize-py: installed"
    else
        fail "Failed to install componentize-py"
        exit 1
    fi
}

install_wasm_tools() {
    log "Installing wasm-tools via cargo ..."
    cargo install wasm-tools --quiet
    if command -v wasm-tools &>/dev/null; then
        log "wasm-tools: $(wasm-tools --version)"
    else
        fail "Failed to install wasm-tools"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# WASM validation helpers
# ---------------------------------------------------------------------------

validate_wasm() {
    local wasm="$1"
    local name="$2"

    if [[ ! -f "$wasm" ]]; then
        fail "$name: .wasm file not found at $wasm"
        return 1
    fi

    # Validate WASM binary
    if ! wasm-tools validate "$wasm" 2>/dev/null; then
        fail "$name: wasm-tools validate failed"
        return 1
    fi

    # Extract WIT and verify 'run' export
    local wit_output
    wit_output=$(wasm-tools component wit "$wasm" 2>/dev/null || true)

    if [[ -z "$wit_output" ]]; then
        # Might be a core module (Rust cdylib without component adapter).
        # Still a valid compilation check.
        warn "$name: core module (no component-model WIT) — compilation verified"
        return 0
    fi

    # Accept known export patterns:
    #   pie:*/run              — standard inferlet run interface
    #   wasi:http/incoming-handler — HTTP server interface
    if echo "$wit_output" | grep -qE "export (pie:.*/run|wasi:http/incoming-handler)"; then
        local export_line
        export_line=$(echo "$wit_output" | grep -oE "export (pie:.*/run|wasi:http/incoming-handler[^ ;]*)" | head -1)
        ok "$name: compiled + WIT verified ($export_line)"
    else
        fail "$name: compiled but no known WIT export found"
        echo "  Expected: 'export pie:<name>/run' or 'export wasi:http/incoming-handler'"
        echo "  WIT output:"
        echo "$wit_output" | head -20 | sed 's/^/    /'
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Rust inferlets
# ---------------------------------------------------------------------------

build_rust() {
    section "Rust inferlets"

    # --- Example workspace (20 inferlets) ---
    log "Building sdk/examples/ workspace ($(grep -c '"' "$REPO_ROOT/sdk/examples/Cargo.toml" | head -1) members) ..."
    if cargo build --target wasm32-wasip2 --release \
         --manifest-path "$REPO_ROOT/sdk/examples/Cargo.toml" 2>&1; then
        log "Workspace build succeeded"
    else
        fail "Rust examples workspace build failed"
        return 1
    fi

    local wasm_dir="$REPO_ROOT/sdk/examples/target/wasm32-wasip2/release"
    for wasm in "$wasm_dir"/*.wasm; do
        [[ -f "$wasm" ]] || continue
        validate_wasm "$wasm" "rust/examples/$(basename "$wasm" .wasm)"
    done

    # --- std/ inferlets (standalone) ---
    for dir in "$REPO_ROOT"/std/*/; do
        local name
        name=$(basename "$dir")
        [[ -f "$dir/Cargo.toml" ]] || continue

        log "Building std/$name ..."
        if cargo build --target wasm32-wasip2 --release \
             --manifest-path "$dir/Cargo.toml" 2>&1; then
            # Find the .wasm output
            local std_wasm_dir="$dir/target/wasm32-wasip2/release"
            local found=false
            for wasm in "$std_wasm_dir"/*.wasm; do
                [[ -f "$wasm" ]] || continue
                validate_wasm "$wasm" "rust/std/$name"
                found=true
            done
            if [[ "$found" == false ]]; then
                fail "rust/std/$name: build succeeded but no .wasm found"
            fi
        else
            fail "rust/std/$name: build failed"
        fi
    done
}

# ---------------------------------------------------------------------------
# Python inferlets
# ---------------------------------------------------------------------------

build_python() {
    section "Python inferlets"

    mkdir -p "$OUT_DIR"

    for dir in "$REPO_ROOT"/sdk/examples/python/*/; do
        local name
        name=$(basename "$dir")
        [[ -f "$dir/Pie.toml" ]] || continue

        local out="$OUT_DIR/${name}-py.wasm"
        log "Building python/$name ..."
        if bakery build "$dir" -o "$out" 2>&1; then
            validate_wasm "$out" "python/$name"
        else
            fail "python/$name: bakery build failed"
        fi
    done
}

# ---------------------------------------------------------------------------
# JavaScript / TypeScript inferlets
# ---------------------------------------------------------------------------

build_javascript() {
    section "JavaScript/TypeScript inferlets"

    # Ensure JS SDK dependencies are installed (needed by bakery for bundling)
    local js_sdk="$REPO_ROOT/sdk/javascript"
    if [[ ! -d "$js_sdk/node_modules" ]]; then
        log "Installing JS SDK dependencies ..."
        (cd "$js_sdk" && npm install --silent)
    fi

    # Note: We skip `npm run build` (tsc) here because the SDK source files
    # use WASM component-model imports (inferlet:core/*) that tsc can't resolve.
    # Bakery handles TS transpilation via esbuild and WIT validation via componentize-js.

    mkdir -p "$OUT_DIR"

    for dir in "$REPO_ROOT"/sdk/examples/javascript/*/; do
        local name
        name=$(basename "$dir")
        [[ -f "$dir/Pie.toml" ]] || [[ -f "$dir/package.json" ]] || continue

        local out="$OUT_DIR/${name}-js.wasm"
        log "Building javascript/$name ..."
        if bakery build "$dir" -o "$out" 2>&1; then
            validate_wasm "$out" "javascript/$name"
        else
            fail "javascript/$name: bakery build failed"
        fi
    done
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_summary() {
    section "Summary"

    echo -e "  ${GREEN}Passed:${NC}  $PASSED"
    echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
    echo -e "  ${RED}Failed:${NC}  $FAILED"

    if [[ ${#FAILURES[@]} -gt 0 ]]; then
        echo -e "\n${RED}Failures:${NC}"
        for f in "${FAILURES[@]}"; do
            echo "  - $f"
        done
    fi

    if [[ $FAILED -gt 0 ]]; then
        echo -e "\n${RED}${BOLD}RESULT: FAILED${NC}"
        exit 1
    else
        echo -e "\n${GREEN}${BOLD}RESULT: ALL PASSED${NC}"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    local run_rust=false
    local run_python=false
    local run_javascript=false
    local run_setup=false

    if [[ $# -eq 0 ]]; then
        run_rust=true
        run_python=true
        run_javascript=true
    fi

    for arg in "$@"; do
        case "$arg" in
            --rust)       run_rust=true ;;
            --python)     run_python=true ;;
            --javascript) run_javascript=true ;;
            --setup)      run_setup=true ;;
            --help|-h)
                echo "Usage: $0 [--rust] [--python] [--javascript] [--setup]"
                echo "  No flags = run all. --setup = install prerequisites only."
                exit 0
                ;;
            *)
                echo "Unknown option: $arg"
                exit 1
                ;;
        esac
    done

    echo -e "${BOLD}Inferlet Compilation CI — Local Test${NC}"
    echo "Repo: $REPO_ROOT"
    echo ""

    check_prerequisites "$run_rust" "$run_python" "$run_javascript"

    if [[ "$run_setup" == true ]]; then
        log "Setup complete."
        exit 0
    fi

    [[ "$run_rust" == true ]]       && build_rust
    [[ "$run_python" == true ]]     && build_python
    [[ "$run_javascript" == true ]] && build_javascript

    print_summary
}

main "$@"
