#!/bin/sh
# gotiny.sh <run dir> <prep dir> <levers K:V,..> [run_g12.py args...]: the Gemma 4 12B tiny program on the
# simulator (the tiny prep's geometry, split-geo) from this branch's kernels, stack-room checked. Env: CSL_SDK (the SDK dir).
# Prep: mk_tiny_g12.py <model dir>, then prep_g12.py on it with the tiny geometry.
T=$(cd $(dirname $0) && pwd); D=$1; P=$(cd $2 && pwd); XP=$3; shift 3
SDK=${CSL_SDK:?set CSL_SDK to the SDK directory (cslc, cs_python)}
mkdir -p $D && python3 $T/mkfull_g12.py $D $P "${PRE:-}" "${OVR:-G2=2,G2W=2}" "$XP" >/dev/null && rm -rf $D/prep && cp -r $P $D/prep && cd $D && rm -rf out
export SINGULARITYENV_CSL_SUPPRESS_SIMFAB_TRACE=1 SINGULARITYENV_OMP_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PAR=$(cat params.txt); . ./geo.env
echo $PAR > params.used
$SDK/cslc --arch=wse3 layout.csl --fabric-dims=$FW,$FH --fabric-offsets=4,1 --params=$PAR -o out --memcpy --channels=1 > c.log 2>&1 || { echo compile failed; grep -m5 -i error -A3 c.log; exit 1; }
CSL_SDK=$SDK python3 $T/../chk_stack.py out > stack.log || { cat stack.log; echo STACK_ROOM_FAIL; exit 1; }
tail -n 1 stack.log
timeout -k 10 ${TMO:-3600} $SDK/cs_python run_g12.py --name out --report r.json --prep prep "$@" > run.log 2>&1
echo rc $?
grep -E '^\{"i"|^\{"launch|^G12|^REP|Error|error|Traceback' run.log | tail -20 | cut -c1-400
