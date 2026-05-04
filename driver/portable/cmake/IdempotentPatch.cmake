# Apply a patch idempotently to the current working directory.
#
# Used as PATCH_COMMAND for vendored llama.cpp. On CI runners with a
# persistent build cache (e.g. our self-hosted GPU runner keeps
# `_deps/llama-src/` between jobs) the source can already be in the
# patched state when ExternalProject re-runs the populate step.
#
# Strategy: probe with `patch -N --dry-run`. The `-N` (`--forward`) flag
# suppresses BSD `patch`'s auto-reverse heuristic — without it, a patched
# tree + closed stdin makes `patch` silently reverse-apply the patch and
# return rc=0, masking real drift. With `-N`:
#
#   * forward dry-run rc=0                                → unpatched,
#                                                           apply for real.
#   * forward fails, reverse dry-run rc=0                  → already
#                                                           patched, skip.
#   * else                                                → real failure
#                                                           (hunk drift
#                                                           after upstream
#                                                           rev bump, or
#                                                           corrupt tree).
#                                                           Fail loudly at
#                                                           configure time
#                                                           with the
#                                                           diagnostic
#                                                           output, rather
#                                                           than silently
#                                                           skipping and
#                                                           surfacing as a
#                                                           missing-symbol
#                                                           compile error
#                                                           ~60 min later.
#
# Required: -DPATCH_FILE=<absolute path to .patch>

if(NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "IdempotentPatch.cmake: -DPATCH_FILE=<...> is required")
endif()

get_filename_component(PATCH_NAME ${PATCH_FILE} NAME)

execute_process(
  COMMAND patch -p1 -N --dry-run -i ${PATCH_FILE}
  RESULT_VARIABLE probe_rc
  OUTPUT_VARIABLE probe_out
  ERROR_VARIABLE  probe_err
  INPUT_FILE      /dev/null)

if(probe_rc EQUAL 0)
  message(STATUS "applying ${PATCH_NAME}")
  execute_process(
    COMMAND patch -p1 -N -i ${PATCH_FILE}
    RESULT_VARIABLE apply_rc
    INPUT_FILE      /dev/null)
  if(NOT apply_rc EQUAL 0)
    message(FATAL_ERROR "${PATCH_NAME}: patch failed with exit ${apply_rc}")
  endif()
else()
  execute_process(
    COMMAND patch -p1 -R --dry-run -i ${PATCH_FILE}
    RESULT_VARIABLE reverse_rc
    OUTPUT_VARIABLE reverse_out
    ERROR_VARIABLE  reverse_err
    INPUT_FILE      /dev/null)
  if(reverse_rc EQUAL 0)
    message(STATUS "${PATCH_NAME} already applied (skipped)")
  else()
    message(FATAL_ERROR
      "${PATCH_NAME}: hunks no longer apply forward and tree is not in "
      "patched state — patch needs to be refreshed against the current "
      "upstream source.\n"
      "----- patch -N --dry-run output (rc=${probe_rc}) -----\n"
      "${probe_out}${probe_err}\n"
      "----- patch -R --dry-run output (rc=${reverse_rc}) -----\n"
      "${reverse_out}${reverse_err}")
  endif()
endif()
