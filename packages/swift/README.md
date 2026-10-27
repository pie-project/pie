# Pie for Swift

In-process Metal engine (`PieServer`), WebSocket/in-process client (`PieClient`), and language components (`PieLanguagePython`, `PieLanguageJavaScript`) for iOS and macOS.

```swift
import PieServer
import PieLanguagePython

struct Prompt: Encodable { let prompt: String }

let model = Bundle.main.url(forResource: "qwen", withExtension: "zt")!
let server = try await PieServer.start(model: model, languages: [.python])
let client = try await server.connect()
let name = try await server.install(contentsOf: Bundle.main.url(forResource: "main", withExtension: "py")!)

let process = try await client.launch(name, input: Prompt(prompt: "Hi!"))
for try await event in process.events {
    switch event {
    case .stdout(let text), .message(let text): print(text)
    case .stderr: break
    case .returned(let value): print("returned", value)
    }
}
await server.shutdown()
```

Pass `listen: "127.0.0.1:8080"` to `PieServer.start` to expose `/v1/ws` and the OpenAI, Anthropic, and Gemini HTTP routes. `PieClient.connect(to:)` connects to a remote `pie serve`.

## Models & build

Import a `.metal.zt` artifact on macOS:

```bash
pie model import Qwen/Qwen3.5-0.8B --deployment qwen35-d0.8b-u4g64-kv-bf16
```

```bash
./build-xcframework.sh                  # ../c → build/PieServerCore.xcframework
../../scripts/build-languages.sh        # language components
swift test
```
