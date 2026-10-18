#!/bin/bash
# Replays a trace and SIGKILLs a server process (and its children) partway:
#   fault.sh <url> <seconds> <pid> <label> <out> [replay args...]
# The client retries a failed request once, whole, as an agent framework
# would; what a system loses is what it makes the client do again.
set -u
URL=$1; AT=$2; VICTIM=$3; LABEL=$4; OUT=$5; shift 5
DIR=$(cd "$(dirname "$0")" && pwd)
( sleep "$AT"
  KIDS=$(pgrep -P "$VICTIM" | tr '\n' ' ')
  kill -9 "$VICTIM" $KIDS 2>/dev/null
  echo "killed $VICTIM ${KIDS:+and $KIDS}at ${AT}s" >&2 ) &
python3 "$DIR/replay.py" run --url "$URL" --label "$LABEL" --out "$OUT" --retries 1 "$@"
