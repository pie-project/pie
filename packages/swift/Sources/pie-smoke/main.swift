// One chat turn through compat-openai, end to end on a Mac:
//   swift run pie-smoke <model.zt | ws://host:port> ["prompt"]
// or, with PIE_SCRIPT=<inferlet.py | inferlet.js>, one run of that script
// inferlet (in-process only) with {"prompt", "max_tokens"} as its input.
// PIE_LISTEN=<host:port> also serves the gateway, and the turn goes through it.

import Foundation
import PieLanguageJavaScript
import PieLanguagePython
import PieServer

struct ChatRequest: Encodable {
    struct Message: Encodable {
        let role: String
        let content: String
    }

    let messages: [Message]
    let maxTokens: Int
    let stream = true
    let chatTemplateKwargs = ["enable_thinking": false]

    enum CodingKeys: String, CodingKey {
        case messages, stream
        case maxTokens = "max_tokens"
        case chatTemplateKwargs = "chat_template_kwargs"
    }
}

struct ScriptInput: Encodable {
    let prompt: String
    let max_tokens: Int
}

struct Chunk: Decodable {
    struct Data: Decodable {
        struct Choice: Decodable {
            struct Delta: Decodable { let content: String? }
            let delta: Delta
        }
        let choices: [Choice]
    }
    let data: Data
}

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
    FileHandle.standardError.write(Data("usage: pie-smoke <model.zt | ws://host:port> [prompt]\n".utf8))
    exit(2)
}
let prompt = arguments.count >= 3 ? arguments[2] : "Explain what a KV cache is in two sentences."
let maxTokens = ProcessInfo.processInfo.environment["MAX_TOKENS"].flatMap(Int.init) ?? 128

let server: PieServer?
let client: PieClient
let clock = ContinuousClock()
let booted = clock.now
if let url = URL(string: arguments[1]), url.scheme == "ws" {
    server = nil
    client = try await PieClient.connect(to: url)
    print("connected to \(url)")
} else {
    let listen = ProcessInfo.processInfo.environment["PIE_LISTEN"]
    server = try await PieServer.start(model: URL(filePath: arguments[1]), listen: listen)
    if listen != nil, let address = server!.listenAddress {
        print("gateway at \(address)")
        client = try await PieClient.connect(to: URL(string: "ws://\(address)")!)
    } else {
        client = try await server!.connect()
    }
    print(String(format: "booted %@ in %.2fs", server!.summary.sku, booted.duration(to: clock.now) / .seconds(1)))
}

if let server, let script = ProcessInfo.processInfo.environment["PIE_SCRIPT"].map({ URL(filePath: $0) }) {
    try await server.install(script.pathExtension == "py" ? .python : .javascript)
    let name = try await server.install(contentsOf: script)
    print(try await client.launch(name, input: ScriptInput(prompt: prompt, max_tokens: maxTokens)).result())
    await client.close()
    await server.shutdown()
    exit(0)
}

let launched = clock.now
let process = try await client.launch(
    "compat-openai",
    input: ChatRequest(messages: [.init(role: "user", content: prompt)], maxTokens: maxTokens)
)
var firstToken: ContinuousClock.Instant?
var tokens = 0
for try await case .message(let json) in process.events {
    guard let text = (try? JSONDecoder().decode(Chunk.self, from: Data(json.utf8)))?.data.choices.first?.delta.content else {
        continue
    }
    firstToken = firstToken ?? clock.now
    tokens += 1
    print(text, terminator: "")
    fflush(stdout)
}
if let firstToken {
    let ttft = launched.duration(to: firstToken) / .seconds(1)
    let decode = firstToken.duration(to: clock.now) / .seconds(1)
    print(String(format: "\nTTFT %.2fs · %.1f tok/s · %d tokens", ttft, Double(tokens - 1) / decode, tokens))
}

await client.close()
await server?.shutdown()
print("shut down")
