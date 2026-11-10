#!/bin/sh
# link.sh <build dir> [<prep params.txt> [extra cslc params K:V,...]]
#
# Links the whole role-split program in <build dir> at the wafer's dims, then checks every PE image's
# stack room (chk_stack.py, >= 768 B). Run it on every build before it goes to the chip: the linker
# does not count the stack. The output is <build dir>/out.
#
# A build of mkfull.py (split-geo: <build dir>/geo.env) is resolved: its dims are geo.env's and it
# takes no shape params; levers go to mkfull.py, not here. An older build takes the prep's shape
# params (prep_q35.py's params.txt; later K:V pairs override earlier ones).
#
# CSL_SDK: the directory holding the SDK's cslc and cs_readelf.
# The build is IEEE f16 (no --fp16-format=bf16): the kernels' f16 arithmetic assumes it.
set -e
: "${CSL_SDK:?set CSL_SDK to the SDK directory (cslc, cs_readelf)}"
D=$1; PP=${2:-}; X=${3:-}
T=$(cd "$(dirname "$0")" && pwd)
if [ -f "$D/geo.env" ]; then
  [ -z "$X" ] || { echo "link.sh: a split-geo build takes its levers in mkfull.py"; exit 1; }
  . "$D/geo.env"; P=$(cat "$D/params.txt")
else
P=$(python3 -c "import sys; d={}; [d.__setitem__(*kv.split(':')) for kv in (open(sys.argv[1]).read().strip()+','+sys.argv[2]).split(',') if kv]; print(','.join(k+':'+v for k,v in d.items()))" "$PP" "$X")
eval "$(python3 -c "
d=dict(kv.split(':') for kv in '$P'.split(',')); i=lambda k:int(d[k])
W=i('UNX')*(i('PX')+1)+i('HC')+1; H=max(i('UNY')*(i('PY')+1),i('HR')); print(f'FW={W+7} FH={H+2} PW={W} PH={H}')")"
fi
cd "$D"
echo "$P" > params.used
echo "rect $PW x $PH (fabric $FW,$FH)"
rm -rf out
SINGULARITYENV_CSL_SUPPRESS_SIMFAB_TRACE=1 "$CSL_SDK/cslc" --arch=wse3 layout.csl --fabric-dims=$FW,$FH \
  --fabric-offsets=4,1 --params="$P" -o out --memcpy --channels=1 > c.log 2>&1 \
  || { grep -m5 -i error c.log; echo LINK_FAIL; exit 1; }
python3 "$T/chk_stack.py" out
