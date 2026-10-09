# pie-shim

C-ABI staticlib that embeds the Pie standalone (controller, gateway and
worker composed in one process, Metal engine) for the iOS app under
`ios/`. The boot path is the one `pie run` uses: the config TOML is a
standalone config exactly as `pie config init` writes it, with
`[server] port` forced to 0 so the OS picks a loopback port.

## ABI v3

The full contract is documented in `src/lib.rs`.

    typedef void (*pie_stream_cb)(int32_t kind, const char* chunk, void* ctx);
    char* pie_ios_run_stream(const char* config_path, const char* wasm_path,
                             const char* version /* nullable */,
                             const char* input_json, uint64_t turn_id,
                             pie_stream_cb cb, void* ctx);
    void pie_ios_cancel(uint64_t turn_id);
    void pie_ios_free(char* s);

| Symbol | Does | Thread |
| --- | --- | --- |
| `pie_ios_run_stream` | Boots the engine on the first call, installs the component once per process, launches it with `input_json`, streams its output to `cb`, returns its return value | Blocks the calling thread for the turn; `cb` runs on that thread |
| `pie_ios_cancel` | Stops the turn named `turn_id`, in flight or not started yet | Any thread; never blocks on the engine |
| `pie_ios_free` | Releases a string `pie_ios_run_stream` returned | Any thread |

`cb` receives each chunk as NUL-terminated UTF-8 with its kind:

| `kind` | Chunk | From the inferlet |
| --- | --- | --- |
| 0 | Reply text, speakable as it arrives | stdout |
| 1 | Reasoning text, never to be spoken | session messages (`inferlet::session::send`) |

The inferlet's stderr goes to the process's stderr, prefixed
`[inferlet stderr] `, which the app mirrors into its log file.

`pie_ios_run_stream` returns one of:

| Result | When |
| --- | --- |
| The inferlet's return value (voice-chat: its JSON) | The turn ran to completion |
| Exactly `PIE CANCELLED` | `pie_ios_cancel` stopped the turn, before or during; text already streamed was delivered through `cb` |
| A string starting with `PIE ERROR: ` | Any failure, a panic on the calling thread included |

## Turn ids and cancellation

The caller picks a `turn_id` per call, unique for the life of the process
(the app uses an increasing counter). `pie_ios_cancel(turn_id)`:

| Turn state | Effect |
| --- | --- |
| Running (launched) | Its process is terminated on the engine, which acknowledges in milliseconds; the call returns `PIE CANCELLED` and the engine is free for the next turn |
| Booting the engine or installing its component (the first turn of the process only) | Takes effect right after that step; nothing is launched |
| Not called yet | The id is remembered (the most recent 64), and the call returns `PIE CANCELLED` at once when it comes, without launching |
| Returned | Nothing |

A cancel only flips the turn's state under a short lock and wakes it, so it
is safe from the main thread while `pie_ios_run_stream` is blocked on a
background queue. A cancel that reaches a turn at the same moment as its
last event still wins: the call returns `PIE CANCELLED`.

## Other guarantees

- The engine boots on the first call and stays warm for the life of the
  process; a failed boot is final until the app relaunches.
- A turn that has not returned after 240 s has its process terminated and
  returns an error.
- NULs inside a chunk or the result are stripped, never truncate it.

## Build

The crate shares the repository's `target/` via `.cargo/config.toml`:

    cd ios/pie-shim
    cargo build --release --target aarch64-apple-ios        # device
    cargo build --release --target aarch64-apple-ios-sim    # simulator

The library lands at `target/<triple>/release/libpie_ios_shim.a` under
the repository root. Engine diagnostics go to stderr; `RUST_LOG`
overrides the default `info,tarpc=warn` filter.

## Test

`tests/turns.rs` drives the same extern functions on a Mac's GPU: a plain
turn, a long turn cancelled after its first reply chunk (must return
within 1 s of the cancel), the turn after it (first chunk within 2 s), a
turn cancelled before it was called, a thinking turn (reasoning chunks
all before the first reply chunk), and a barge-in: a greedy reply
cancelled partway, then the next turn of that conversation with the
partial reply in its transcript, which must resume from the prefix the
cancelled turn published and answer exactly as it does after a reply that
completed. It needs a Metal model artifact and a built voice-chat
inferlet, and skips with a note without them:

    cd examples && cargo build --release --target wasm32-wasip2 -p voice-chat
    cd ../ios/pie-shim
    PIE_SHIM_TEST_MODEL=/path/to/qwen3.5-0.8b.metal.zt \
    PIE_SHIM_TEST_WASM=$PWD/../../examples/target/wasm32-wasip2/release/voice_chat.wasm \
        cargo test --release -- --nocapture
