# Pie for Swift

Two libraries:

- **`PieClient`** speaks pie's client protocol (MessagePack frames,
  crates/client-api). It talks to a remote `pie serve` over WebSocket, or to
  the runtime a `PieServer` runs in this process.
- **`PieServer`** runs pie inside an iOS or macOS app: the runtime, the Metal
  engine and the inferlet sandbox in-process, booted through
  `runtime::embed` (the same path the browser build uses). Inferlets run
  under wasmtime's Pulley interpreter on iOS, where an app may not JIT.

```swift
import PieServer

let server = try await PieServer.start(model: Bundle.main.url(forResource: "qwen", withExtension: "zt")!)
let client = try await server.connect()           // a PieClient; frames never leave the process
let process = try await client.launch("compat-openai", input: [
    "messages": [["role": "user", "content": "Hi!"]],
    "stream": true,
])
for try await event in process.events { print(event) }   // message / stdout / return / error
await server.shutdown()
```

The built-in inferlets (`compat-openai`, ...) are registered at boot; your
own install from bytes (`server.install(contentsOf:)`) and launch by the
`name@version` that returns. `PieClient.connect(to: URL(string: "ws://host:8080")!)`
reaches a `pie serve` with the same API.

## Models

`PieServer` takes a `.metal.zt` artifact, which a Mac with a Metal build of
pie produces; the shaders compile on the device at boot, so an artifact
imported on a Mac loads on an iPhone:

```bash
pie model import Qwen/Qwen3.5-0.8B --sku qwen35-d0.8b-u4g64-kv-bf16   # 441 MB, 4-bit
```

`PieServer.Config` holds the engine budgets (KV pages, forward tokens and
lanes, the share of the GPU working set), sized for a phone by default.

## Build

```bash
./build-xcframework.sh        # the Rust core → build/PieServerCore.xcframework (iOS, iOS simulator, macOS; arm64)
swift run -c release pie-smoke <model.zt> "prompt"   # end to end on a Mac
```

`server/` is the Rust core (`pie-server-swift`, a workspace member) behind
`server/include/pie_server.h`; anything that speaks C can link it the same way.

## Limits

- The runtime boots once per process. `shutdown()` releases the engine and
  its memory, but the app cannot start another server afterwards.
- In the iOS simulator an app builds and runs, but the engine refuses to boot:
  the simulator's GPU shares no memory with the host. Inference needs a device.
- Linking `PieServer` adds about 57 MB to an app's executable (about 16 MB
  compressed); the model ships or downloads separately.
