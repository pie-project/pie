import Foundation

/// A client of a pie server (a remote `pie serve`, or a `PieServer` in this
/// process), speaking crates/client-api's MessagePack protocol.
public final class PieClient: @unchecked Sendable {
    public struct Failure: Error, CustomStringConvertible {
        public let description: String
        public init(description: String) { self.description = description }
    }

    private let transport: PieTransport
    private let lock = NSLock()
    private var nextCorrId: UInt32 = 0
    private var pending: [UInt32: CheckedContinuation<(Bool, String), Error>] = [:]
    private var streams: [String: AsyncThrowingStream<PieProcess.Event, Error>.Continuation] = [:]
    private var orphans: [String: [PieProcess.Event]] = [:]
    private var closed: Error?
    private var reader: Task<Void, Never>?

    /// Connects to a `pie serve` at `ws://host:port`.
    public static func connect(to url: URL, identity: String? = "default/swift") async throws -> PieClient {
        try await connect(over: WebSocketTransport(url: url, identity: identity))
    }

    /// Runs the protocol over `transport`; resolves once the server answers a ping.
    public static func connect(over transport: PieTransport) async throws -> PieClient {
        let client = PieClient(transport: transport)
        try await client.ping()
        return client
    }

    private init(transport: PieTransport) {
        self.transport = transport
        reader = Task { [weak self] in
            do {
                for try await frame in transport.frames {
                    guard let self else { return }
                    if let message = try? MessagePack.decode(frame) { self.dispatch(message) }
                }
                self?.fail(Failure(description: "the connection closed"))
            } catch {
                self?.fail(error)
            }
        }
    }

    deinit {
        reader?.cancel()
        transport.close()
    }

    public func close() {
        transport.close()
        fail(Failure(description: "the connection is closed"))
    }

    /// Launches an installed inferlet (`name` or `name@version`) with `input`,
    /// a JSON value its `main` receives. Its events stream from the result.
    public func launch(_ inferlet: String, input: Any = [String: Any](), captureOutputs: Bool = true) async throws -> PieProcess {
        let json = try JSONSerialization.data(withJSONObject: input, options: [.fragmentsAllowed])
        let (ok, result) = try await request([
            "type": .string("launch_process"),
            "inferlet": .string(inferlet),
            "input": .string(String(decoding: json, as: UTF8.self)),
            "capture_outputs": .bool(captureOutputs),
        ])
        guard ok else { throw Failure(description: "launch \(inferlet): \(result)") }
        return PieProcess(client: self, id: result, events: stream(for: result))
    }

    /// Round-trips a ping; true when the server answered ok.
    @discardableResult
    public func ping() async throws -> Bool {
        try await request(["type": .string("ping")]).0
    }

    func signal(_ processId: String, _ message: String) async throws {
        try await transport.send(MessagePack.map([
            "type": .string("signal_process"),
            "process_id": .string(processId),
            "message": .string(message),
        ]).encoded())
    }

    func terminate(_ processId: String) async throws {
        _ = try await request([
            "type": .string("terminate_process"),
            "process_id": .string(processId),
        ])
    }

    // MARK: - Wire

    private func request(_ fields: [String: MessagePack]) async throws -> (Bool, String) {
        let corrId: UInt32 = lock.withLock {
            nextCorrId &+= 1
            return nextCorrId
        }
        var fields = fields
        fields["corr_id"] = .int(Int64(corrId))
        let frame = MessagePack.map(fields).encoded()
        return try await withCheckedThrowingContinuation { continuation in
            let failed: Error? = lock.withLock {
                if let closed { return closed }
                pending[corrId] = continuation
                return nil
            }
            if let failed { continuation.resume(throwing: failed); return }
            Task {
                do { try await self.transport.send(frame) } catch {
                    self.lock.withLock { self.pending.removeValue(forKey: corrId) }?.resume(throwing: error)
                }
            }
        }
    }

    private func dispatch(_ message: MessagePack) {
        switch message["type"]?.string {
        case "response":
            guard let corrId = message["corr_id"]?.int else { return }
            let continuation = lock.withLock { pending.removeValue(forKey: UInt32(corrId)) }
            continuation?.resume(returning: (message["ok"]?.bool ?? false, message["result"]?.string ?? ""))
        case "process_event":
            guard let id = message["process_id"]?.string else { return }
            let event = PieProcess.Event(
                name: message["event"]?.string ?? "",
                value: message["value"]?.string ?? ""
            )
            let continuation: AsyncThrowingStream<PieProcess.Event, Error>.Continuation? = lock.withLock {
                if let continuation = streams[id] { return continuation }
                orphans[id, default: []].append(event)
                return nil
            }
            guard let continuation else { return }
            continuation.yield(event)
            if event.isTerminal {
                continuation.finish()
                lock.withLock { _ = streams.removeValue(forKey: id) }
            }
        default:
            break
        }
    }

    /// Replays events that arrived before the launch response.
    private func stream(for id: String) -> AsyncThrowingStream<PieProcess.Event, Error> {
        AsyncThrowingStream { continuation in
            let early: [PieProcess.Event] = lock.withLock {
                streams[id] = continuation
                return orphans.removeValue(forKey: id) ?? []
            }
            for event in early {
                continuation.yield(event)
                if event.isTerminal {
                    continuation.finish()
                    lock.withLock { _ = streams.removeValue(forKey: id) }
                    return
                }
            }
        }
    }

    private func fail(_ error: Error) {
        let (requests, open) = lock.withLock {
            closed = closed ?? error
            defer { pending.removeAll(); streams.removeAll() }
            return (Array(pending.values), Array(streams.values))
        }
        for continuation in requests { continuation.resume(throwing: error) }
        for continuation in open { continuation.finish(throwing: error) }
    }
}

/// A launched inferlet. `events` yields what it prints and sends, and ends
/// after its `return` or `error` event.
public struct PieProcess: Sendable {
    public struct Event: Sendable, CustomStringConvertible {
        /// `stdout`, `stderr`, `message` (a `session::send`), `return` or `error`.
        public let name: String
        public let value: String

        public var isTerminal: Bool { name == "return" || name == "error" }
        public var description: String { "\(name): \(value)" }
    }

    let client: PieClient
    public let id: String
    public let events: AsyncThrowingStream<Event, Error>

    /// Delivers `message` to the inferlet's `session::receive`.
    public func signal(_ message: String) async throws {
        try await client.signal(id, message)
    }

    public func terminate() async throws {
        try await client.terminate(id)
    }

    /// Drains the events and returns what `main` returned; throws its error.
    public func result() async throws -> String {
        for try await event in events {
            switch event.name {
            case "return": return event.value
            case "error": throw PieClient.Failure(description: event.value)
            default: continue
            }
        }
        throw PieClient.Failure(description: "the process ended without a result")
    }
}
