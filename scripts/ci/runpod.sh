#!/usr/bin/env bash
# The RunPod pod that is the CUDA runner: start it before the GPU job, stop
# it after. Needs RUNPOD_API_KEY and RUNPOD_POD_ID; in Actions, stop also
# needs GH_TOKEN to see whether another run is using the pod.
#
#   runpod.sh start | stop | status
set -euo pipefail
api="https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID:?}"
call() { curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer ${RUNPOD_API_KEY:?}" -H "Content-Type: application/json" "$api$2"; }
status() { call GET "" | jq -r '"\(.desiredStatus) gpu=\(.machine.gpuTypeId // .gpu // "-") uptime=\(.runtime.uptimeInSeconds // 0)s"'; }
case "${1:?start|stop|status}" in
  status) status ;;
  start)
    s=$(status); echo "$s"
    case "$s" in RUNNING*) exit 0 ;; esac
    call POST /start >/dev/null
    # A stopped pod resumes only when its host has the GPU free; say so
    # rather than leaving the job to queue for a runner that never comes.
    for _ in $(seq 1 30); do
      s=$(status); echo "$s"
      case "$s" in RUNNING*) exit 0 ;; esac
      sleep 10
    done
    echo "pod ${RUNPOD_POD_ID} did not reach RUNNING" >&2; exit 1 ;;
  stop)
    # Another run's GPU job may be queued for or on the pod; that run's own
    # stop step turns it off.
    if [[ -n "${GITHUB_RUN_ID:-}" ]]; then
      for run in $(gh api "repos/$GITHUB_REPOSITORY/actions/runs?per_page=30" \
          -q ".workflow_runs[] | select(.id != $GITHUB_RUN_ID and (.status == \"queued\" or .status == \"in_progress\")) | .id"); do
        if gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run/jobs" \
            -q '.jobs[] | select(.name == "cuda: serve on hardware" and .status != "completed") | .id' | grep -q .; then
          echo "run $run is on the pod; leaving it up"; exit 0
        fi
      done
    fi
    call POST /stop >/dev/null; echo "pod ${RUNPOD_POD_ID} stopped" ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
