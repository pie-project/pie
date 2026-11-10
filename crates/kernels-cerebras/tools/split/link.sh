#!/bin/sh
# link.sh <build dir>
#
# Links the whole wafer program in <build dir> (mkfull.py's: its dims geo.env's, every param a literal) at the
# wafer's dims, checks every PE image's stack room (chk_stack.py, >= 768 B), and puts the artifact the engine runs
# (<build dir>/geo: program.json, host.json, bindings.json, rcfg.npy, rcfg_n.npy) in <build dir>/out/artifact. Run
# it on every build before it goes to the chip: the linker does not count the stack. The output is <build dir>/out.
#
# CSL_SDK: the directory holding the SDK's cslc and cs_readelf.
# The build is IEEE f16 (no --fp16-format=bf16): the kernels' f16 arithmetic assumes it.
set -e
: "${CSL_SDK:?set CSL_SDK to the SDK directory (cslc, cs_readelf)}"
D=$1
T=$(cd "$(dirname "$0")" && pwd)
[ -f "$D/geo.env" ] || { echo "link.sh: $D is not a mkfull.py build (no geo.env)"; exit 1; }
. "$D/geo.env"
cd "$D"
echo "rect $PW x $PH (fabric $FW,$FH)"
rm -rf out
SINGULARITYENV_CSL_SUPPRESS_SIMFAB_TRACE=1 "$CSL_SDK/cslc" --arch=wse3 layout.csl --fabric-dims=$FW,$FH \
  --fabric-offsets=4,1 -o out --memcpy --channels=1 > c.log 2>&1 \
  || { grep -m5 -i error c.log; echo LINK_FAIL; exit 1; }
python3 "$T/chk_stack.py" out
mkdir -p out/artifact
cp geo/program.json geo/host.json geo/bindings.json geo/rcfg.npy geo/rcfg_n.npy out/artifact/
