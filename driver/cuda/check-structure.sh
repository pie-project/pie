#!/bin/sh

# Phase 8 structural gate. This script is intentionally POSIX sh and needs only
# find, awk, sort, and grep. It may be run from the repository root or here.

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd) || exit 1
cd "$script_dir" || exit 1
LC_ALL=C
export LC_ALL

fail=0

fail_with() {
    printf '%s\n' "::error::$1" >&2
    fail=1
}

# Strip CMake comments outside quoted strings, then emit literal repository
# paths. Fetched dependency paths contain variables and are intentionally not
# treated as files in this checkout.
cmake_paths=$(
    awk '
        function without_comment(s,    i, c, quoted, escaped, out) {
            quoted = 0
            escaped = 0
            out = ""
            for (i = 1; i <= length(s); ++i) {
                c = substr(s, i, 1)
                if (c == "#" && !quoted) break
                out = out c
                if (c == "\"" && !escaped) quoted = !quoted
                if (c == "\\" && !escaped) {
                    escaped = 1
                } else {
                    escaped = 0
                }
            }
            return out
        }
        {
            line = without_comment($0)
            gsub(/\$\{CMAKE_CURRENT_SOURCE_DIR\}\//, "", line)
            gsub(/[()";,]/, " ", line)
            count = split(line, field, /[[:space:]]+/)
            for (i = 1; i <= count; ++i) {
                if (field[i] ~ /^(cmake|src|tests|third_party)\//) {
                    print field[i]
                }
            }
        }
    ' CMakeLists.txt | sort -u
)

# Generated/build output is never source inventory. Everything else under
# tests/ (sources, support headers, fixtures, scripts, and their README) must be
# named explicitly in CMake.
test_files=$(
    find tests \
        \( -type d \( -name __pycache__ -o -name build -o -name generated \
            -o -name CMakeFiles -o -name _deps \) -prune \) -o \
        \( -type f ! -name .DS_Store -print \) |
        sort
)

old_ifs=$IFS
IFS='
'

unrepresented=
for path in $test_files; do
    if ! printf '%s\n' "$cmake_paths" | grep -Fqx "$path"; then
        unrepresented="${unrepresented}  ${path}
"
    fi
done
if [ -n "$unrepresented" ]; then
    fail_with "Files under tests/ are not represented explicitly in CMakeLists.txt:"
    printf '%s' "$unrepresented" >&2
fi

missing=
for path in $cmake_paths; do
    if [ ! -e "$path" ]; then
        missing="${missing}  ${path}
"
    fi
done
if [ -n "$missing" ]; then
    fail_with "CMakeLists.txt references nonexistent repository paths:"
    printf '%s' "$missing" >&2
fi

IFS=$old_ifs

# Parse complete CMake commands (including multiline conditions) and track
# nested if/elseif/else blocks. Runtime filesystem probes are allowed, but an
# EXISTS-dependent branch may not declare a target or add sources.
guarded_hits=$(
    awk '
        function without_comment(s,    i, c, quoted, escaped, out) {
            quoted = 0
            escaped = 0
            out = ""
            for (i = 1; i <= length(s); ++i) {
                c = substr(s, i, 1)
                if (c == "#" && !quoted) break
                out = out c
                if (c == "\"" && !escaped) quoted = !quoted
                if (c == "\\" && !escaped) {
                    escaped = 1
                } else {
                    escaped = 0
                }
            }
            return out
        }
        function paren_delta(s,    i, c, quoted, escaped, value) {
            quoted = 0
            escaped = 0
            value = 0
            for (i = 1; i <= length(s); ++i) {
                c = substr(s, i, 1)
                if (c == "\"" && !escaped) quoted = !quoted
                if (!quoted && c == "(") ++value
                if (!quoted && c == ")") --value
                if (c == "\\" && !escaped) {
                    escaped = 1
                } else {
                    escaped = 0
                }
            }
            return value
        }
        function has_exists(s) {
            s = toupper(s)
            return s ~ /(^|[^A-Z0-9_])EXISTS([^A-Z0-9_]|$)/
        }
        function command_name(s,    name) {
            name = s
            sub(/^[[:space:]]*/, "", name)
            sub(/[[:space:]]*\(.*/, "", name)
            return tolower(name)
        }
        function process_command(s, line_number,    name, parent_guard) {
            name = command_name(s)
            if (name == "if") {
                parent_guard = depth > 0 ? guarded[depth] : 0
                ++depth
                inherited[depth] = parent_guard
                exists_block[depth] = has_exists(s)
                guarded[depth] = inherited[depth] || exists_block[depth]
                return
            }
            if (name == "elseif") {
                if (depth > 0 && has_exists(s)) exists_block[depth] = 1
                if (depth > 0) {
                    guarded[depth] = inherited[depth] || exists_block[depth]
                }
                return
            }
            if (name == "else") {
                if (depth > 0) {
                    guarded[depth] = inherited[depth] || exists_block[depth]
                }
                return
            }
            if (name == "endif") {
                if (depth > 0) {
                    delete guarded[depth]
                    delete inherited[depth]
                    delete exists_block[depth]
                    --depth
                }
                return
            }
            if (depth > 0 && guarded[depth] &&
                name ~ /^(add_executable|add_library|add_custom_target|target_sources|set_source_files_properties)$/) {
                printf "%d: %s\n", line_number, name
            }
        }
        {
            clean = without_comment($0)
            if (collecting) {
                command = command " " clean
                balance += paren_delta(clean)
                if (balance <= 0) {
                    process_command(command, command_line)
                    collecting = 0
                    command = ""
                }
                next
            }
            if (clean ~ /^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(/) {
                collecting = 1
                command = clean
                command_line = NR
                balance = paren_delta(clean)
                if (balance <= 0) {
                    process_command(command, command_line)
                    collecting = 0
                    command = ""
                }
            }
        }
    ' CMakeLists.txt
)
if [ -n "$guarded_hits" ]; then
    fail_with "CMakeLists.txt has EXISTS-guarded targets or sources:"
    printf '%s\n' "$guarded_hits" >&2
fi

# Read only real preprocessor include directives, not comments or arbitrary
# text. Leading ./ and ../ components are normalized before applying the exact
# module rules.
scan_forbidden_includes() {
    scan_dir=$1
    forbidden=$2
    find "$scan_dir" \
        \( -type d \( -name build -o -name generated -o -name CMakeFiles \
            -o -name _deps \) -prune \) -o \
        \( -type f -exec awk -v forbidden="$forbidden" '
            /^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]/ {
                include = $0
                sub(/^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]/, "", include)
                sub(/[>"].*$/, "", include)
                while (sub(/^\.\.?\//, "", include)) {}
                if (include ~ "^(" forbidden ")/") {
                    printf "%s:%d: %s\n", FILENAME, FNR, include
                }
            }
        ' {} + \) |
        sort
}

kernels_upward=$(scan_forbidden_includes src/kernels 'model|batch|pipeline|loader|store')
if [ -n "$kernels_upward" ]; then
    fail_with "kernels/ includes model, batch, pipeline, loader, or store:"
    printf '%s\n' "$kernels_upward" >&2
fi

ops_upward=$(scan_forbidden_includes src/ops 'model|batch|pipeline|loader')
if [ -n "$ops_upward" ]; then
    fail_with "ops/ includes model, batch, pipeline, or loader:"
    printf '%s\n' "$ops_upward" >&2
fi

loader_upward=$(scan_forbidden_includes src/loader 'batch|pipeline')
if [ -n "$loader_upward" ]; then
    fail_with "loader/ includes batch or pipeline:"
    printf '%s\n' "$loader_upward" >&2
fi

if [ "$fail" -ne 0 ]; then
    printf '%s\n' "driver/cuda structure gate FAILED." >&2
    exit 1
fi

printf '%s\n' "driver/cuda structure gate PASSED."
