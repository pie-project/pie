#!/usr/bin/env bash
# compare.sh BASE CAND: two dump.sh outputs. Shard plans and coverage lines
# by diff; CSL byte for byte; StableHLO that differs in bytes through
# canon.py (equal dataflow up to emission order and SSA names). Exits 1 on
# any difference; each one is left in CAND/<name>.diff to read.
set -uo pipefail
base=$1
cand=$2
here=$(dirname "$0")
status=0

if diff "$base/shard.txt" "$cand/shard.txt" >"$cand/shard.diff"; then
    echo "shard: identical"
else
    echo "shard: DIFFERS ($(grep -c '^[<>]' "$cand/shard.diff") lines; see $cand/shard.diff)"
    status=1
fi

for d in xla cerebras-phase cerebras-table cerebras-whole; do
    n=$(find "$base/$d" -type f | wc -l)
    diff -q "$base/$d.txt" "$cand/$d.txt" >/dev/null ||
        { echo "$d: coverage lines differ (diff $base/$d.txt $cand/$d.txt)"; status=1; }
    if diff -rq "$base/$d" "$cand/$d" >"$cand/$d.diff"; then
        echo "$d: identical ($n files)"
    elif [ "$d" = xla ]; then
        printf "xla: "
        python3 "$here/canon.py" "$base/$d" "$cand/$d" "$cand/$d.diff" | tail -n1
        [ "${PIPESTATUS[0]}" = 0 ] || status=1
    else
        echo "$d: DIFFERS ($(wc -l <"$cand/$d.diff") of $n files; see $cand/$d.diff)"
        status=1
    fi
done
exit $status
