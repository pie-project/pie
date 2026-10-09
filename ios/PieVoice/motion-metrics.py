#!/usr/bin/env python3
"""motion-metrics.py OUT_DIR - how each step of a motion tour moved.

For every step the tour marked, decodes the recording at its native frame
rate for the 0.9 s after the mark, shrinks each frame to grayscale, and
measures the change between consecutive frames. Reports:

  frames   frames recorded in the window (the Simulator records on change)
  moving   frames whose change is above noise
  snap     the largest single frame's share of all the change: near 1.0
           means the step happened in one frame (a snap); a smooth
           animation spreads it, typically under 0.35
  span_ms  time from the first to the last moving frame
  gaps     recording gaps over 34 ms while something was moving (dropped
           frames, or the app stalling)

Usage: python3 motion-metrics.py ios/PieVoice/build/motion
"""
import os
import subprocess
import sys

import numpy as np

W, H = 90, 196
WINDOW = 0.9
NOISE = 0.6  # mean absolute difference (0-255) treated as no change


_STAMPS = {}


def all_stamps(video):
    """Every frame's presentation time in the recording, read once."""
    if video not in _STAMPS:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=pts_time",
                              "-of", "csv=p=0", video], capture_output=True, text=True).stdout.split()
        _STAMPS[video] = [float(t.strip(",")) for t in out if t.strip(",")]
    return _STAMPS[video]


def frames(video, start, length):
    """The recorded frames from start to start+length, and their times."""
    times = [t for t in all_stamps(video) if start <= t < start + length]
    if not times:
        return np.zeros((0, H, W), dtype=np.float32), []
    first, count = times[0], len(times)
    cmd = ["ffmpeg", "-v", "error", "-ss", "%.4f" % first, "-i", video, "-frames:v", str(count),
           "-vsync", "0", "-vf", "scale=%d:%d,format=gray" % (W, H), "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    count = min(count, len(raw) // (W * H))
    arr = np.frombuffer(raw[: count * W * H], dtype=np.uint8).reshape(count, H, W).astype(np.float32)
    return arr, times[:count]


def main(out):
    start = float(open(os.path.join(out, "rec-start.txt")).read())
    marks = []
    for line in open(os.path.join(out, "markers.txt")):
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "MOTION" and parts[1].replace(".", "", 1).isdigit():
            marks.append((float(parts[1]) - start, parts[2]))
    video = os.path.join(out, "tour.mp4")
    print("%-30s %6s %6s %6s %8s  %s" % ("step", "frames", "moving", "snap", "span_ms", "gaps>34ms"))
    for i, (t, name) in enumerate(marks):
        length = WINDOW
        if i + 1 < len(marks):
            length = min(WINDOW, max(0.2, marks[i + 1][0] - t))
        arr, times = frames(video, max(0.0, t - 0.03), length)
        if len(arr) < 2:
            print("%-30s %6d      -      -        -" % (name, len(arr)))
            continue
        diffs = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2))
        moving = diffs > NOISE
        total = diffs[moving].sum()
        snap = float(diffs.max() / total) if total > 0 else 0.0
        idx = np.nonzero(moving)[0]
        span = (times[idx[-1] + 1] - times[idx[0]]) * 1000 if len(idx) and len(times) > idx[-1] + 1 else 0
        gaps = []
        if len(idx):
            lo, hi = idx[0], idx[-1] + 1
            for a, b in zip(times[lo:hi], times[lo + 1:hi + 1]):
                if b - a > 0.034:
                    gaps.append(int((b - a) * 1000))
        print("%-30s %6d %6d %6.2f %8.0f  %s" % (name, len(arr), int(moving.sum()), snap, span, gaps[:6]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/motion")
