import Foundation
import Synchronization
import Testing
@testable import PieClient

/// A server scripted per request: `script` sees each client frame and
/// answers through `push`.
final class MockServer: PieTransport {
    let frames: AsyncThrowingStream<Data, any Error>
    private let continuation: AsyncThrowingStream<Data, any Error>.Continuation
    private let script: @Sendable (MessagePack, MockServer) -> Void
    let received = Mutex<[MessagePack]>([])

    init(_ script: @escaping @Sendable (MessagePack, MockServer) -> Void) {
        (frames, continuation) = AsyncThrowingStream.makeStream(of: Data.self)
        self.script = script
    }

    func send(_ frame: Data) async throws {
        let message = try MessagePack.decode(frame)
        received.withLock { $0.append(message) }
        script(message, self)
    }

    func close() {
        continuation.finish()
    }

    func push(_ fields: [String: MessagePack]) {
        continuation.yield(MessagePack.map(fields).encoded())
    }

    func respond(to message: MessagePack, ok: Bool = true, result: String = "") {
        push(["type": .string("response"), "corr_id": message["corr_id"] ?? .int(0), "ok": .bool(ok), "result": .string(result)])
    }

    func event(_ process: String, _ event: String, _ value: String) {
        push(["type": .string("process_event"), "process_id": .string(process), "event": .string(event), "value": .string(value)])
    }

    func sent(_ type: String) -> [MessagePack] {
        received.withLock { $0.filter { $0["type"]?.string == type } }
    }
}

/// Answers pings; `launch` handles each launch.
func server(launch: @escaping @Sendable (MessagePack, MockServer) -> Void = { _, _ in }) -> MockServer {
    MockServer { message, server in
        switch message["type"]?.string {
        case "ping", "terminate_process": server.respond(to: message)
        case "launch_process": launch(message, server)
        default: break
        }
    }
}

struct Input: Encodable {
    let prompt: String
    let maxTokens: Int
}

struct PieClientTests {
    @Test func launchesWithEncodedInputAndStreamsEvents() async throws {
        let mock = server { message, server in
            server.respond(to: message, result: "p1")
            server.event("p1", "stdout", "hello")
            server.event("p1", "message", "{\"delta\":\"x\"}")
            server.event("p1", "return", "done")
        }
        let client = try await PieClient(transport: mock)
        let process = try await client.launch("echo", input: Input(prompt: "hi", maxTokens: 3))

        var events: [PieProcess.Event] = []
        for try await event in process.events { events.append(event) }
        #expect(events == [.stdout("hello"), .message("{\"delta\":\"x\"}"), .returned("done")])

        let launch = try #require(mock.sent("launch_process").first)
        #expect(launch["inferlet"]?.string == "echo")
        let input = try JSONSerialization.jsonObject(with: Data(try #require(launch["input"]?.string).utf8)) as? [String: Any]
        #expect(input?["prompt"] as? String == "hi")
    }

    @Test func keepsEventsThatBeatTheLaunchResponse() async throws {
        let mock = server { message, server in
            server.event("p2", "stdout", "early")
            server.event("p2", "return", "ok")
            server.respond(to: message, result: "p2")
        }
        let client = try await PieClient(transport: mock)
        let process = try await client.launch("echo")
        #expect(try await process.result() == "ok")
    }

    @Test func throwsTheInferletsError() async throws {
        let mock = server { message, server in
            server.respond(to: message, result: "p3")
            server.event("p3", "error", "boom")
        }
        let client = try await PieClient(transport: mock)
        let process = try await client.launch("echo")
        await #expect(throws: PieError.processFailed("boom")) { try await process.result() }
    }

    @Test func throwsARefusedLaunch() async throws {
        let mock = server { message, server in server.respond(to: message, ok: false, result: "no such inferlet") }
        let client = try await PieClient(transport: mock)
        await #expect(throws: PieError.requestFailed("no such inferlet")) { try await client.launch("missing") }
    }

    @Test func failsPendingRequestsWhenTheConnectionCloses() async throws {
        let mock = server { _, server in server.close() }
        let client = try await PieClient(transport: mock)
        await #expect(throws: PieError.connectionClosed) { try await client.launch("echo") }
        await #expect(throws: PieError.connectionClosed) { try await client.ping() }
    }

    @Test func cancellingTheEventsTerminatesTheInferlet() async throws {
        let mock = server { message, server in
            server.respond(to: message, result: "p4")
            server.event("p4", "stdout", "working")
        }
        let client = try await PieClient(transport: mock)
        let process = try await client.launch("loop")
        let consumer = Task {
            for try await _ in process.events {}
        }
        try await Task.sleep(for: .milliseconds(50))
        consumer.cancel()
        for _ in 0..<100 where mock.sent("terminate_process").isEmpty {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(mock.sent("terminate_process").first?["process_id"]?.string == "p4")
    }

    @Test func refusesToConnectWithoutAPingAnswer() async throws {
        let mock = MockServer { message, server in server.respond(to: message, ok: false, result: "busy") }
        await #expect(throws: PieError.requestFailed("busy")) { try await PieClient(transport: mock) }
    }
}
