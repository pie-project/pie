#!/bin/bash
# motion-tour.sh — record every animated interaction of Pie Voice in the
# Simulator with real touches, on the demo backend, and cut the recording
# into one frame sheet per step.
#
#   bash ios/PieVoice/motion-tour.sh            # build, record, cut
#   SIM=<udid> OUT=<dir> bash ios/PieVoice/motion-tour.sh
#
# Needs the simulator shim:
#   (cd ios/pie-shim && CARGO_TARGET_DIR=../../target cargo build --release --target aarch64-apple-ios-sim)
# Writes $OUT/tour.mp4, $OUT/markers.txt, $OUT/steps/NN-name.png (frames at
# 20 fps across each step, 10 to a row) and $OUT/steps/NN-name.mp4.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SIM=${SIM:-0B5628C3-A89C-46CC-B2EB-E204A9916C64}   # iPhone 17 Pro, iOS 26.0
OUT=${OUT:-$HERE/build/motion}
TEST=${TEST:-testMotionTour}
DERIVED=$HERE/build/SimDerivedData
mkdir -p "$OUT/steps"
rm -f "$OUT"/steps/*

( cd "$HERE" && xcodegen --use-cache >/dev/null ) || { echo "xcodegen failed"; exit 1; }
xcodebuild -project "$HERE/PieVoice.xcodeproj" -scheme PieVoice -configuration ${CONFIGURATION:-Release} \
  -destination "id=$SIM" -derivedDataPath "$DERIVED" ONLY_ACTIVE_ARCH=YES ARCHS=arm64 build-for-testing > "$OUT/build.log" 2>&1 \
  || { grep -E "error:" "$OUT/build.log" | head -20; echo "build failed: $OUT/build.log"; exit 1; }

xcrun simctl boot "$SIM" 2>/dev/null; xcrun simctl bootstatus "$SIM" -b >/dev/null 2>&1
xcrun simctl privacy "$SIM" grant microphone org.pie-project.voice 2>/dev/null
xcrun simctl terminate "$SIM" org.pie-project.voice 2>/dev/null

rm -f "$OUT/tour.mp4" "$OUT/rec.log"
xcrun simctl io "$SIM" recordVideo --codec=h264 --force "$OUT/tour.mp4" 2> "$OUT/rec.log" &
REC=$!
for _ in $(seq 1 100); do grep -q "Recording started" "$OUT/rec.log" 2>/dev/null && break; sleep 0.1; done
python3 -c 'import time; print("%.3f" % time.time())' > "$OUT/rec-start.txt"

xcodebuild -project "$HERE/PieVoice.xcodeproj" -scheme PieVoice -configuration ${CONFIGURATION:-Release} \
  -destination "id=$SIM" -derivedDataPath "$DERIVED" ONLY_ACTIVE_ARCH=YES ARCHS=arm64 \
  -only-testing:PieVoiceUITests/MotionTourUITests/$TEST test-without-building > "$OUT/test.log" 2>&1
TEST_EXIT=$?
sleep 1
kill -INT "$REC"; wait "$REC" 2>/dev/null
grep -E "^MOTION" "$OUT/test.log" | sed 's/^.*MOTION/MOTION/' > "$OUT/markers.txt"
grep -h "MOTION-SKIP" "$OUT/test.log" | sort -u

python3 - "$OUT" <<'PY'
import os, subprocess, sys
out = sys.argv[1]
start = float(open(os.path.join(out, "rec-start.txt")).read())
marks = []
for line in open(os.path.join(out, "markers.txt")):
    parts = line.split()
    if len(parts) >= 3 and parts[0] == "MOTION":
        marks.append((float(parts[1]) - start, parts[2]))
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            os.path.join(out, "tour.mp4")], capture_output=True, text=True).stdout.strip() or 0)
for i, (t, name) in enumerate(marks):
    end = marks[i + 1][0] if i + 1 < len(marks) else dur
    begin = max(0.0, t - 0.15)
    length = max(0.3, min(end - begin, 3.0))
    base = os.path.join(out, "steps", "%02d-%s" % (i, name))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % begin, "-t", "%.3f" % length,
                    "-i", os.path.join(out, "tour.mp4"), "-vf", "fps=20,scale=240:-1,tile=10x6:padding=4:color=white",
                    "-frames:v", "1", base + ".png"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % begin, "-t", "%.3f" % length,
                    "-i", os.path.join(out, "tour.mp4"), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", base + ".mp4"])
    print("%6.2f s  %-32s %4.2f s" % (t, name, length))
PY
echo "test exit $TEST_EXIT; recording $OUT/tour.mp4; sheets $OUT/steps/"
