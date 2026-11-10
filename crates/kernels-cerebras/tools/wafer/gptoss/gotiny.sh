#!/bin/sh
# gotiny.sh <run dir> <prep dir> <plan dir> <levers K:V,..> [run_gptoss.py args...]: the gpt-oss tiny program on the
# simulator (the tiny prep's geometry, wafer-compile) from this branch's kernels, stack-room checked. Env: CSL_SDK (the SDK dir).
T=$(cd $(dirname $0) && pwd); D=$1; P=$(cd $2 && pwd); PL=$(cd $3 && pwd); XP=$4; shift 4
SDK=${CSL_SDK:?set CSL_SDK to the SDK directory (cslc, cs_python)}
mkdir -p $D && python3 $T/build_gptoss.py $D $PL/plan.json "$XP" >/dev/null && rm -rf $D/prep && cp -r $P $D/prep && cd $D && rm -rf out
export SINGULARITYENV_CSL_SUPPRESS_SIMFAB_TRACE=1 SINGULARITYENV_OMP_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
. ./dims.env
[ -z "$SIMCONFIG" ] || cp $SIMCONFIG simconfig.json   # optional simulator config (threads)
$SDK/cslc --arch=wse3 layout.csl --fabric-dims=$FW,$FH --fabric-offsets=4,1 -o out --memcpy --channels=1 > c.log 2>&1 || { echo compile failed; grep -m5 -i error -A3 c.log; exit 1; }
CSL_SDK=$SDK python3 $T/../chk_stack.py out > stack.log || { cat stack.log; echo STACK_ROOM_FAIL; exit 1; }
tail -n 1 stack.log
timeout -k 10 ${TMO:-1800} $SDK/cs_python run_gptoss.py --name out --report r.json --prep prep "$@" > run.log 2>&1
echo rc $?
grep -E '^\{"i"|^REP|^GPTOSS|^TIMES|stall|Error|error|Traceback' run.log | tail -12 | cut -c1-400
