// One chat turn through compat-openai, end to end on a Mac:
//   swift run pie-smoke <model.zt | ws://host:port> ["prompt"]

import Foundation
import PieServer

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write(Data("usage: pie-smoke <model.zt | ws://host:port> [prompt]\n".utf8))
    exit(2)
}
let prompt = args.count >= 3 ? args[2] : "Explain what a KV cache is in two sentences."

var server: PieServer?
let client: PieClient
let started = Date()
if args[1].hasPrefix("ws://") {
    client = try await PieClient.connect(to: URL(string: args[1])!)
    print("connected to \(args[1])")
} else {
    server = try await PieServer.start(model: URL(fileURLWithPath: args[1]))
    client = try await server!.connect()
    print(String(format: "booted %@ in %.2fs", server!.summary["sku"] as? String ?? "?", Date().timeIntervalSince(started)))
}

let turn = Date()
let process = try await client.launch("compat-openai", input: [
    "messages": [["role": "user", "content": prompt]],
    "max_tokens": Int(ProcessInfo.processInfo.environment["MAX_TOKENS"] ?? "128") ?? 128,
    "stream": true,
    "chat_template_kwargs": ["enable_thinking": false],
])
var first: Date?
var pieces = 0
for try await event in process.events {
    if event.name == "error" { print("\n[error] \(event.value)") }
    guard event.name == "message",
          let frame = try? JSONSerialization.jsonObject(with: Data(event.value.utf8)) as? [String: Any],
          let data = frame["data"] as? [String: Any],
          let choices = data["choices"] as? [[String: Any]],
          let text = (choices.first?["delta"] as? [String: Any])?["content"] as? String
    else { continue }
    if first == nil { first = Date() }
    pieces += 1
    print(text, terminator: "")
    fflush(stdout)
}
if let first {
    let rate = Double(max(pieces - 1, 0)) / Date().timeIntervalSince(first)
    print(String(format: "\nTTFT %.2fs · %.1f tok/s · %d pieces", first.timeIntervalSince(turn), rate, pieces))
}

client.close()
await server?.shutdown()
print("shut down")
