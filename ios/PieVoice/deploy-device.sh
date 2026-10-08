#!/bin/bash
# deploy-device.sh — from "iPhone plugged in" to "Pie Voice running with the
# engine ready", in one command. Re-runnable; every step is incremental.
#
#   bash ios/PieVoice/deploy-device.sh            # build, sign, install, launch, verify
#   bash ios/PieVoice/deploy-device.sh --no-rust  # skip the cargo steps (Swift-only change)
#   bash ios/PieVoice/deploy-device.sh --log      # just pull the on-device console log
#
# Environment overrides:
#   DEVELOPMENT_TEAM  Apple team id      (default: Aarush's Personal Team)
#   DEVICE            CoreDevice id/UDID (default: the first connected iPhone)
#   CONFIGURATION     Release | Debug    (default: Release — Debug ggml is ~1 tok/s)
#
# Why re-sign every time: a free Apple ID's provisioning profile lasts 7 days.
# Each run mints a fresh one, so "it worked last week" never becomes "the app
# won't open" — as long as it is run again within the week.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HERE="$ROOT/ios/PieVoice"
BUNDLE=org.pie-project.voice
TEAM=${DEVELOPMENT_TEAM:-JQ8RG44463}
CONFIGURATION=${CONFIGURATION:-Release}
DERIVED="$HERE/build/DerivedData"
APP="$DERIVED/Build/Products/$CONFIGURATION-iphoneos/PieVoice.app"
MODELS_DIR=${MODELS_DIR:-$HOME/random/pie-models}
BUNDLED_MODEL=Qwen3-0.6B-Q4_K_M.gguf
LOG_LOCAL="$HERE/build/pie-console.log"

DO_RUST=1
ONLY_LOG=0
for arg in "$@"; do
  case "$arg" in
    --no-rust) DO_RUST=0 ;;
    --log) ONLY_LOG=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

bold() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mFAILED:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Find the phone ────────────────────────────────────────────────────────
find_device() {
  local json="$HERE/build/devices.json"
  mkdir -p "$HERE/build"
  xcrun devicectl list devices --json-output "$json" >/dev/null 2>&1 || return 1
  python3 - "$json" "${DEVICE:-}" <<'PY'
import json, sys
devices = json.load(open(sys.argv[1]))["result"]["devices"]
want = sys.argv[2]
candidates = []
for d in devices:
    hw, conn, props = d["hardwareProperties"], d["connectionProperties"], d["deviceProperties"]
    if hw.get("platform") != "iOS" or hw.get("deviceType") != "iPhone":
        continue
    ident = d["identifier"]
    if want and want not in (ident, hw.get("udid"), props.get("name")):
        continue
    if conn.get("tunnelState") != "connected":
        continue
    if props.get("developerModeStatus") not in (None, "enabled"):
        print(f"# {props.get('name')} is connected but Developer Mode is off", file=sys.stderr)
        continue
    wired = 0 if conn.get("transportType") == "wired" else 1
    candidates.append((wired, ident, hw.get("udid"), props.get("name"), hw.get("marketingName", ""), props.get("osVersionNumber", "")))
if candidates:
    c = sorted(candidates)[0]
    print(c[1], c[2], c[3], c[4], c[5])
PY
}

wait_for_device() {
  bold "Looking for the iPhone"
  local waited=0 line=""
  while :; do
    line=$(find_device)
    [ -n "$line" ] && break
    if [ $waited -eq 0 ]; then
      echo "No connected iPhone yet. Plug it in with a cable, unlock it, and tap"
      echo "'Trust' if asked. Developer Mode must be on (Settings > Privacy &"
      echo "Security > Developer Mode). Waiting up to 5 minutes…"
    fi
    [ $waited -ge 300 ] && die "no connected iPhone after 5 minutes"
    sleep 5; waited=$((waited+5))
  done
  DEVICE_ID=$(echo "$line" | awk '{print $1}')
  DEVICE_UDID=$(echo "$line" | awk '{print $2}')
  echo "Found: ${line#* * }  (CoreDevice $DEVICE_ID)"
}

# ── 6. Pull and interpret the on-device console log ──────────────────────────
pull_log() {
  rm -f "$LOG_LOCAL"
  xcrun devicectl device copy from --device "$DEVICE_ID" \
    --domain-type appDataContainer --domain-identifier "$BUNDLE" \
    --source Documents/pie-console.log --destination "$LOG_LOCAL" >/dev/null 2>&1
  [ -f "$LOG_LOCAL" ]
}

verdict_from_log() {
  # Only the most recent launch counts.
  local last
  last=$(awk '/=== PieVoice launch/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$LOG_LOCAL")
  if echo "$last" | grep -q '\[warmup\] engine boot complete'; then
    echo READY
  elif echo "$last" | grep -qE 'panicked at|fatal runtime error|engine boot FAILED|PIE ERROR'; then
    echo FAILED
  else
    echo PENDING
  fi
  printf '%s' "$last" > "$LOG_LOCAL.last"
}

if [ $ONLY_LOG -eq 1 ]; then
  wait_for_device
  pull_log || die "could not pull Documents/pie-console.log (is the app installed and launched once?)"
  xcrun devicectl device copy from --device "$DEVICE_ID" \
    --domain-type appDataContainer --domain-identifier "$BUNDLE" \
    --source Documents/pie-console.log.prev --destination "$LOG_LOCAL.prev" >/dev/null 2>&1 || true
  verdict_from_log
  echo "--- last launch ---"; cat "$LOG_LOCAL.last"
  echo "(full log: $LOG_LOCAL; rotated older log, if any: $LOG_LOCAL.prev)"
  exit 0
fi

wait_for_device

# ── 2. Rust: engine shim for the device + the inferlet ───────────────────────
if [ $DO_RUST -eq 1 ]; then
  bold "Building the Pie engine shim for arm64 iOS (incremental)"
  rustup target list --installed | grep -q '^aarch64-apple-ios$' || rustup target add aarch64-apple-ios
  rustup target list --installed | grep -q '^wasm32-wasip2$'     || rustup target add wasm32-wasip2
  ( cd "$ROOT/ios/pie-shim" && cargo build --release --target aarch64-apple-ios 2>&1 | tail -3 ) \
    || die "cargo build of ios/pie-shim failed"
  [ -f "$ROOT/ios/pie-shim/target/aarch64-apple-ios/release/libpie_ios_shim.a" ] \
    || die "shim staticlib missing after build"

  bold "Building the voice-chat inferlet (wasm32-wasip2)"
  ( cd "$ROOT/inferlets/voice-chat" && cargo build --release --target wasm32-wasip2 2>&1 | tail -2 ) \
    || die "cargo build of inferlets/voice-chat failed"
  mkdir -p "$HERE/Resources"
  cp "$ROOT/inferlets/voice-chat/target/wasm32-wasip2/release/voice_chat.wasm" "$HERE/Resources/"
  cp "$ROOT/inferlets/voice-chat/Pie.toml" "$HERE/Resources/voice-chat-Pie.toml"
fi

# ── 3. Bundle contents ───────────────────────────────────────────────────────
bold "Checking bundle resources"
mkdir -p "$HERE/Resources/qwen3-gguf"
if [ ! -f "$HERE/Resources/qwen3-gguf/$BUNDLED_MODEL" ]; then
  for src in "$ROOT/target/qwen3-gguf/$BUNDLED_MODEL" "$MODELS_DIR/$BUNDLED_MODEL"; do
    if [ -f "$src" ]; then
      echo "staging model from $src"
      ln -f "$src" "$HERE/Resources/qwen3-gguf/$BUNDLED_MODEL" 2>/dev/null \
        || cp "$src" "$HERE/Resources/qwen3-gguf/$BUNDLED_MODEL"
      break
    fi
  done
fi
[ -f "$HERE/Resources/qwen3-gguf/$BUNDLED_MODEL" ] || die "model $BUNDLED_MODEL not found (looked in target/qwen3-gguf and $MODELS_DIR)"
[ -f "$HERE/Resources/voice_chat.wasm" ] || die "Resources/voice_chat.wasm missing — run without --no-rust once"
[ -f "$HERE/Resources/sample-question-1.wav" ] || bash "$ROOT/ios/voice-app/make-samples.sh" "$HERE/Resources"
for f in voice_chat.wasm voice-chat-Pie.toml "qwen3-gguf/$BUNDLED_MODEL" sample-question-1.wav; do
  printf '  %-40s %s\n' "$f" "$(du -h "$HERE/Resources/$f" | cut -f1)"
done

# ── 4. Xcode project + signed build ──────────────────────────────────────────
if [ ! -d "$HERE/PieVoice.xcodeproj" ] || [ "$HERE/project.yml" -nt "$HERE/PieVoice.xcodeproj/project.pbxproj" ]; then
  bold "Generating the Xcode project"
  command -v xcodegen >/dev/null || die "xcodegen not installed (brew install xcodegen)"
  ( cd "$HERE" && xcodegen 2>&1 | tail -1 ) || die "xcodegen failed"
fi

bold "Building + signing PieVoice ($CONFIGURATION, team $TEAM)"
BUILD_LOG="$HERE/build/xcodebuild.log"
xcodebuild -project "$HERE/PieVoice.xcodeproj" -scheme PieVoice \
  -configuration "$CONFIGURATION" -destination 'generic/platform=iOS' \
  -derivedDataPath "$DERIVED" \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic \
  build > "$BUILD_LOG" 2>&1
if ! grep -q 'BUILD SUCCEEDED' "$BUILD_LOG"; then
  grep -E 'error:|error ' "$BUILD_LOG" | head -20
  die "xcodebuild failed — full log: $BUILD_LOG"
fi
[ -d "$APP" ] || die "built app not found at $APP"

bold "Verifying the signature"
codesign -dvv "$APP" 2>&1 | grep -E '^(Authority|Identifier|TeamIdentifier)' | head -4
PROFILE="$APP/embedded.mobileprovision"
[ -f "$PROFILE" ] || die "no embedded.mobileprovision — signing did not happen"
EXPIRES=$(security cms -D -i "$PROFILE" 2>/dev/null | plutil -extract ExpirationDate raw -o - - 2>/dev/null)
echo "profile expires: $EXPIRES"
if ! security cms -D -i "$PROFILE" 2>/dev/null | plutil -extract ProvisionedDevices json -o - - 2>/dev/null | grep -q "$DEVICE_UDID"; then
  die "profile does not include this iPhone ($DEVICE_UDID) — open the project in Xcode once, pick the team, and press Run"
fi

# ── 5. Install + launch ──────────────────────────────────────────────────────
bold "Installing on the iPhone ($(du -sh "$APP" | cut -f1) — a minute or two)"
installed=0
for attempt in 1 2 3; do
  out=$(xcrun devicectl device install app --device "$DEVICE_ID" "$APP" --timeout 900 2>&1); rc=$?
  if [ $rc -eq 0 ] && grep -qiE 'installed|installationURL' <<<"$out"; then
    installed=1; break
  fi
  echo "install attempt $attempt failed (exit $rc): $(tail -3 <<<"$out")"
  echo "retrying — keep the phone unlocked"
  sleep 5
done
[ $installed -eq 1 ] || die "install failed three times. Is the phone unlocked? Is Developer Mode on? Try again."

# How many launches the phone's log already holds: a verdict only counts
# once a NEW launch block appears, so a refused launch can't be mistaken
# for last week's success.
launches_before=0
if pull_log; then launches_before=$(grep -c '=== PieVoice launch' "$LOG_LOCAL" || true); fi

bold "Launching"
out=$(xcrun devicectl device process launch --terminate-existing --device "$DEVICE_ID" "$BUNDLE" 2>&1); rc=$?
if [ $rc -ne 0 ]; then
  echo "$out" | tail -5
  die "launch failed (exit $rc). Unlock the phone; if it says the developer is not trusted, open Settings > General > VPN & Device Management and trust the certificate, then re-run."
fi
echo "$out" | grep -iE 'launched' | head -1

# ── 6. Verify from the phone's own log ───────────────────────────────────────
bold "Waiting for the engine to come up (model load ~10-30 s on first launch)"
verdict=PENDING; waited=0
while [ $waited -lt 180 ]; do
  sleep 10; waited=$((waited+10))
  if pull_log; then
    launches_now=$(grep -c '=== PieVoice launch' "$LOG_LOCAL" || true)
    if [ "$launches_now" -gt "$launches_before" ]; then
      verdict=$(verdict_from_log)
      [ "$verdict" != PENDING ] && break
    fi
  fi
  printf '  %3ds…\n' "$waited"
done

echo
case "$verdict" in
  READY)
    printf '\033[32mREADY\033[0m — the engine booted and the model is loaded. Talk to it.\n'
    grep -E '^\[launch\]|\[wasmtime\]|\[warmup\]' "$LOG_LOCAL.last" | tail -6 ;;
  FAILED)
    printf '\033[31mENGINE FAILED\033[0m on the phone. Log of the last launch:\n'
    tail -40 "$LOG_LOCAL.last" ;;
  *)
    echo "No verdict after ${waited}s. Log of the last launch so far:"
    [ -f "$LOG_LOCAL.last" ] && tail -20 "$LOG_LOCAL.last" ;;
esac
echo
echo "full log: $LOG_LOCAL   (re-pull any time: bash ios/PieVoice/deploy-device.sh --log)"
[ "$verdict" = READY ]
