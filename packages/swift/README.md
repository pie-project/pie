# Pie for Swift

Libraries:

- **`PieClient`** speaks pie's client protocol (MessagePack frames,
  crates/client-api). It talks to a remote `pie serve` over WebSocket, or to
  the runtime a `PieServer` runs in this process.
- **`PieServer`** runs pie inside an iOS or macOS app: the runtime, the Metal
  engine and the inferlet sandbox in-process, booted by the same worker
  `pie serve` runs (`worker::Server`). Inferlets run under wasmtime's Pulley
  interpreter on iOS, where an app may not JIT.
- **`PieLanguagePython`**, **`PieLanguageJavaScript`** add the components
  script inferlets (`x.py`, `x.js`) run in:

  ```swift
  import PieLanguagePython

  let server = try await PieServer.start(model: model, languages: [.python])
  let name = try await server.install(contentsOf: Bundle.main.url(forResource: "main", withExtension: "py")!)
  ```

```swift
import PieServer

struct Prompt: Encodable { let prompt: String }

let server = try await PieServer.start(model: Bundle.main.url(forResource: "qwen", withExtension: "zt")!)
let client = try await server.connect()           // frames never leave the process
let process = try await client.launch("my-inferlet", input: Prompt(prompt: "Hi!"))
for try await event in process.events {
    switch event {
    case .stdout(let text), .message(let text): print(text)
    case .stderr: break
    case .returned(let value): print("returned", value)
    }
}
await server.shutdown()
```

`events` ends after `.returned` and throws `PieError.processFailed` when the
inferlet fails; cancelling its iteration terminates the inferlet. The
built-in inferlets (`compat-openai`, ...) are registered at boot; install
your own from bytes (`server.install(contentsOf:)`) and launch them by the
`name@version` that returns. `PieClient.connect(to:)` reaches a
`pie serve` at `ws://host:port` with the same API.

`PieServer.start(model:listen: "127.0.0.1:8080")` also serves pie's gateway
from the app, as `pie serve` would: the WebSocket at `/v1/ws` and the
OpenAI-, Anthropic- and Gemini-compatible HTTP routes (`/v1/chat/completions`,
...), for an SDK in the app or a client on the same network. `listenAddress`
says where it bound (with port 0, the port the OS picked).

## Models

`PieServer` takes a `.metal.zt` artifact, which a Mac with a Metal build of
pie produces; the shaders compile on the device at boot, so an artifact
imported on a Mac loads on an iPhone:

```bash
pie model import Qwen/Qwen3.5-0.8B --sku qwen35-d0.8b-u4g64-kv-bf16   # 441 MB, 4-bit
```

`PieServer.Configuration` holds the engine budgets (KV pages, forward tokens and
lanes, the share of the GPU working set), sized for a phone by default.

## Build

```bash
./build-xcframework.sh        # the Rust core → build/PieServerCore.xcframework (iOS, iOS simulator, macOS; arm64)
../../scripts/build-languages.sh                      # the language components → Sources/PieLanguage*/Resources
swift test                                            # protocol and configuration, no GPU
swift run -c release pie-smoke <model.zt> "prompt"   # end to end on a Mac (or ws://host:port)
PIE_SCRIPT=../../examples/quickstart-py/main.py swift run -c release pie-smoke <model.zt> "prompt"
PIE_LISTEN=127.0.0.1:8080 swift run -c release pie-smoke <model.zt> "prompt"   # the turn through its own gateway
```

`core/` is the Rust core (`pie-swift-core`, a workspace member) behind
`core/include/pie_server.h`; anything that speaks C can link it the same way.

## Limits

- The runtime boots once per process. `shutdown()` releases the engine and
  its memory, but the app cannot start another server afterwards.
- iOS closes an app's listening sockets when it moves to the background, so
  `listen` serves while the app is in the foreground.
- In the iOS simulator an app builds and runs, but the engine refuses to boot:
  the simulator's GPU shares no memory with the host. Inference needs a device.
- Linking `PieServer` adds about 57 MB to an app's executable (about 16 MB
  compressed); the Python component adds about 38 MB (14 MB compressed); the
  model ships or downloads separately.
