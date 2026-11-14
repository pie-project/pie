#!/bin/bash
# stream/sim.sh <program dir> <work dir> [run.py args]: compiles a stream program (wafer-compile stream's
# output: layout.csl, its kernels and printed regions) with programs/wafer/ops in the SDK container and runs
# tools/wafer/stream/run.py on the fabric simulator, one sim at a time machine-wide (waits while another
# simulator is busy). bf16 builds unless fabric.txt says otherwise. CONLY=1: compile only; SKIPC=1: run the
# compiled program kept from before (KEEP=1 keeps it). SDK: $SDK.
set -e
P=$1; D=$2; shift 2
H=$(cd "$(dirname "$0")" && pwd); OPS=$H/../../../programs/wafer/ops
SDK=${SDK:-/cra-1900/ing-cloud/pie-bringup-20261027/sdk-2.10.1}
SIF=$(ls $SDK/*.sif)
read W HH B16 < $P/fabric.txt
FLAG="--fp16-format=bf16"; [ "$B16" = 0 ] && FLAG=""
rm -rf $D/src; [ -n "$SKIPC" ] || rm -rf $D/out; mkdir -p $D/src/ops $D/tmp
cp $P/*.csl $D/src/; cp $OPS/*.csl $D/src/ops/; cp $H/run.py $D/src/
if [ -z "$SKIPC" ]; then
singularity exec --bind=/cra-1900 --bind=$D/tmp:/tmp -C --pwd=$D/src $SIF cslc --arch=wse3 layout.csl \
  --fabric-dims=$((W+7)),$((HH+2)) --fabric-offsets=4,1 --memcpy --channels=${CH:-1} $FLAG -o $D/out > $D/compile.log 2>&1 \
  || { grep -m12 -iE "error|panic" $D/compile.log; exit 1; }
fi
[ -n "$CONLY" ] && { echo compiled; exit 0; }
busy() { for p in $(ls /proc | grep -E '^[0-9]+$'); do case $(readlink /proc/$p/exe 2>/dev/null) in /cb/*) ;; *) continue;; esac
  a=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null || echo 0); sleep 2
  b=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null || echo 0); [ "$b" -gt "$((a+50))" ] && return 0; done; return 1; }
while busy; do sleep 60; done
export SINGULARITYENV_OMP_NUM_THREADS=8
[ -n "$TRACE" ] || export SINGULARITYENV_CSL_SUPPRESS_SIMFAB_TRACE=1 CSL_SUPPRESS_SIMFAB_TRACE=1
# (a simulator's fatal error leaves run.py waiting: the watchdog ends it)
( singularity exec --bind=/cra-1900 --bind=$D/tmp:/tmp -C --pwd=$D/src $SIF timeout -k 10 ${SIMT:-36000} python run.py $D/out $P "$@" > $D/run.log 2>&1 || echo "rc $?" >> $D/run.log ) &
RUN=$!
while kill -0 $RUN 2>/dev/null; do
  if grep -q "Stopping due to fatal" $D/src/sim.log 2>/dev/null; then
    sleep 20; echo "simulator fatal: $(grep -m1 'Stopping due to fatal' $D/src/sim.log | cut -c1-300)" >> $D/run.log
    pkill -P $RUN 2>/dev/null
    break
  fi
  sleep 30
done
wait $RUN 2>/dev/null || true
grep -h "cycles=" $D/src/sim.log 2>/dev/null | tail -1 >> $D/run.log || true
grep -E "^STREAM|Error|error|Traceback|fatal" $D/run.log | tail -8
[ -n "$KEEP" ] || rm -rf $D/out $D/src/simfab_traces
