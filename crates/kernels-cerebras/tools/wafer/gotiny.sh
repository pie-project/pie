#!/bin/sh
# gotiny.sh <run dir> <prep dir> <plan dir> <levers K:V,..> [runner args...]: a tiny program of a family with a bare
# runner (Gemma 4, Gemma 4 12B, gpt-oss) on the simulator from this branch's kernels, stack-room checked: build.py
# (which picks the plan's runner), cslc, chk_stack.py, then the runner on the prep. Env: CSL_SDK (the SDK dir),
# TMO (the run's timeout, s; 3600), SIMCONFIG (an optional simulator config: threads).
# Prep: <family>/mk_tiny_<family>.py <model dir>, then <family>/prep_<family>.py on it for the tiny plan.
T=$(cd $(dirname $0) && pwd); D=$1; P=$(cd $2 && pwd); PL=$(cd $3 && pwd); XP=$4; shift 4
SDK=${CSL_SDK:?set CSL_SDK to the SDK directory (cslc, cs_python)}
mkdir -p $D && python3 $T/build.py $D $PL/plan.json "$XP" >/dev/null && rm -rf $D/prep && cp -r $P $D/prep && cd $D && rm -rf out
RUN=$(ls run_*.py 2>/dev/null | head -1)
[ -n "$RUN" ] || { echo "the plan's family has no bare runner (run the engine's artifact_sim)"; exit 1; }
export SINGULARITYENV_CSL_SUPPRESS_SIMFAB_TRACE=1 SINGULARITYENV_OMP_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
. ./dims.env
[ -z "$SIMCONFIG" ] || cp $SIMCONFIG simconfig.json
$SDK/cslc --arch=wse3 layout.csl --fabric-dims=$FW,$FH --fabric-offsets=4,1 -o out --memcpy --channels=1 > c.log 2>&1 || { echo compile failed; grep -m5 -i error -A3 c.log; exit 1; }
CSL_SDK=$SDK python3 $T/chk_stack.py out > stack.log || { cat stack.log; echo STACK_ROOM_FAIL; exit 1; }
tail -n 1 stack.log
timeout -k 10 ${TMO:-3600} $SDK/cs_python $RUN --name out --report r.json --prep prep "$@" > run.log 2>&1
echo rc $?
grep -E '^\{"i"|^\{"launch|^GEMMA|^GPTOSS|^PROBE|^REP|^TIMES|stall|Error|error|Traceback' run.log | tail -20 | cut -c1-400
