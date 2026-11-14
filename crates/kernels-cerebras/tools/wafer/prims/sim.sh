#!/bin/bash
# prims/sim.sh <prim> <work dir> <fabric w,h> <cslc params> <bf16|ieee> [run.py args after the compiled dir]
# Compiles prims/<prim>/layout.csl (with programs/wafer/ops) in the SDK container and runs prims/<prim>/run.py
# on the fabric simulator, one sim at a time machine-wide (waits while another simulator is busy). SDK: $SDK
# CSL_DIR: a directory of generated modules (a printed region) copied beside the layout.
# (default the bring-up's sdk-2.10.1). The compiled program is removed afterwards; logs stay in <work dir>.
set -e
P=$1; D=$2; FD=$3; PARAMS=$4; FMT=$5; shift 5
H=$(cd "$(dirname "$0")" && pwd); OPS=$H/../../../programs/wafer/ops
SDK=${SDK:-/cra-1900/ing-cloud/pie-bringup-20261027/sdk-2.10.1}
SIF=$(ls $SDK/*.sif)
rm -rf $D/src $D/out; mkdir -p $D/src/ops $D/tmp
cp $H/$P/*.csl $H/$P/run.py $D/src/; cp $OPS/*.csl $D/src/ops/
[ -n "$CSL_DIR" ] && cp $CSL_DIR/*.csl $D/src/
FLAG="--fp16-format=bf16"; [ "$FMT" = ieee ] && FLAG=""
singularity exec --bind=/cra-1900 --bind=$D/tmp:/tmp -C --pwd=$D/src $SIF cslc --arch=wse3 layout.csl \
  --fabric-dims=$FD --fabric-offsets=4,1 --params=$PARAMS --memcpy --channels=1 $FLAG -o $D/out > $D/compile.log 2>&1 \
  || { grep -m8 -iE "error|panic" $D/compile.log; exit 1; }
[ -n "$CONLY" ] && { echo compiled; exit 0; }
# one simulator at a time: wait while any SDK-container process (a simulator's python, a wafer-bench) burns CPU
busy() { for p in $(ls /proc | grep -E '^[0-9]+$'); do case $(readlink /proc/$p/exe 2>/dev/null) in /cb/*) ;; *) continue;; esac
  a=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null || echo 0); sleep 2
  b=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null || echo 0); [ "$b" -gt "$((a+50))" ] && return 0; done; return 1; }
while busy; do sleep 60; done
export SINGULARITYENV_CSL_SUPPRESS_SIMFAB_TRACE=1 CSL_SUPPRESS_SIMFAB_TRACE=1 SINGULARITYENV_OMP_NUM_THREADS=8
singularity exec --bind=/cra-1900 --bind=$D/tmp:/tmp -C --pwd=$D/src $SIF timeout -k 10 ${SIMT:-3600} python run.py $D/out "$@" > $D/run.log 2>&1 || echo "rc $?" >> $D/run.log
grep -h "cycles=" $D/src/sim.log 2>/dev/null | tail -1 >> $D/run.log || true
grep -E "^PRIM|Error|error|Traceback" $D/run.log | tail -5
[ -n "$KEEP" ] || rm -rf $D/out $D/src/simfab_traces
