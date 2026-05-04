# Apply a patch idempotently to the current working directory.
#
# Used as PATCH_COMMAND for vendored llama.cpp. On CI runners with a
# persistent build cache (e.g. our self-hosted GPU runner keeps
# `_deps/llama-src/` between jobs) the source can already be in the
# patched state when ExternalProject re-runs the populate step —
# `patch -N` then exits 1 ("Reversed or previously applied") and
# breaks the build.
#
# Strategy: probe with `patch --dry-run`. If the forward apply would
# succeed cleanly, run it for real. Otherwise the tree is already in
# the patched state and we skip silently. Either way, the build
# proceeds with a correctly-patched source. Real patch failures (a
# stale hunk that no longer applies forward AND isn't already applied)
# fail loudly here at configure time rather than ~60 min later as a
# missing-symbol compile error.
#
# Required: -DPATCH_FILE=<absolute path to .patch>

if(NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "IdempotentPatch.cmake: -DPATCH_FILE=<...> is required")
endif()

get_filename_component(PATCH_NAME ${PATCH_FILE} NAME)

execute_process(
  COMMAND patch -p1 --dry-run --silent -i ${PATCH_FILE}
  RESULT_VARIABLE dry_run_rc
  OUTPUT_QUIET ERROR_QUIET)

if(dry_run_rc EQUAL 0)
  message(STATUS "applying ${PATCH_NAME}")
  execute_process(
    COMMAND patch -p1 -i ${PATCH_FILE}
    RESULT_VARIABLE apply_rc)
  if(NOT apply_rc EQUAL 0)
    message(FATAL_ERROR "${PATCH_NAME}: patch failed with exit ${apply_rc}")
  endif()
else()
  message(STATUS "${PATCH_NAME} already applied (skipped)")
endif()
