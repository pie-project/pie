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
    case .file(let name, let data): try data.write(to: URL.documentsDirectory.appending(path: name ?? "out.bin"))
    case .returned(let value): print("returned", value)
    }
}
await server.shutdown()
```

Pass `listen: "127.0.0.1:8080"` to `PieServer.start` to expose `/v1/ws` and the OpenAI, Anthropic, and Gemini HTTP routes. `PieClient.connect(to:)` connects to a remote `pie serve`.

## Handles

A handle is state the client owns between processes: a working set, a matcher, or media. A launch moves it in, and it cannot be used until it comes back, so fork first to keep one.

```swift
let turn = try await client.workingSet()
let picture = try await client.frames()              // frames(data) uploads encoded pictures
let process = try await client.launch(
    name,
    input: ["ws": turn.marker("ws")],
    handles: ["ws": turn],
    returns: ["ws": turn, "picture": picture]
)
_ = try await process.result()

let png = try await picture.read(format: "png")
try await turn.setProps(["priority": 3, "deadline": nil] as [String: Int?])  // nil removes a prop
let branch = try await turn.fork()
let shape = try await client.matcher(jsonSchema: schema)  // or regex:, ebnf:, or none for any JSON
let stat = try await turn.stat()                     // JSON
try await branch.close()

let kept = try await turn.save("s42/turn7")          // tokens and bytes kept
let again = try await client.loadSnapshot("s42/turn7")
let snapshots = try await client.listSnapshots(prefix: "s42/")
try await client.deleteSnapshot("s42/turn7")
```

A refusal throws `PieError.handleFailed(code, reason)`, whose `code` is a `HandleErrorCode` (`.unknown`, `.dead`, `.quota`, `.inUse`, `.invalid`), or nil when the server named none. Props merge for the worker policy to read while the handle rests; sent while a process has it, they land with it.

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
