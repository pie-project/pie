#!/bin/bash
# Synthesise the bundled sample utterances with macOS `say`.
#
# These drive the "Sample" input source, which feeds recordings through
# the same recogniser and controller as the microphone. Turns 2 and 3 are
# follow-ups: they only make sense if the conversation state survived, so
# playing all three is an end-to-end check of the KV-snapshot path with
# no human in the loop.
#
#   bash ios/voice-app/make-samples.sh <output-dir>
set -euo pipefail

OUT=${1:?usage: make-samples.sh <output-dir>}
VOICE=${SAMPLE_VOICE:-Samantha}
# 16 kHz mono signed 16-bit LE — what SFSpeechURLRecognitionRequest wants.
FORMAT="LEI16@16000"

mkdir -p "$OUT"

say_to() {
  local name=$1
  local text=$2
  if ! say -v "$VOICE" -o "$OUT/$name.wav" --data-format="$FORMAT" "$text" 2>/dev/null; then
    # Fall back to the default voice if $VOICE isn't installed.
    say -o "$OUT/$name.wav" --data-format="$FORMAT" "$text"
  fi
}

say_to sample-question-1 "What is one good reason to run a language model on a phone instead of in the cloud?"
say_to sample-question-2 "Can you say that more simply?"
say_to sample-question-3 "What is the hardest part of doing that?"

echo "samples written to $OUT"
