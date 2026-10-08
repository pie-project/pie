import Foundation

/// A client of a pie server: a remote `pie serve`, or a `PieServer` in this
/// process.
public actor PieClient {
    private let transport: any PieTransport
    private var nextCorrelationID: UInt32 = 0
    private var pending: [UInt32: Slot] = [:]
    private var processes: [String: AsyncThrowingStream<PieProcess.Event, any Error>.Continuation] = [:]
    /// Events that arrived before their process's launch response.
    private var early: [String: [(event: String, value: String)]] = [:]
    private var failure: (any Error)?
    private var reader: Task<Void, Never>?

    private enum Slot {
        case awaiting(CheckedContinuation<String, any Error>?)
        case answered(Result<String, any Error>)
    }

    /// Connects to a `pie serve` at `ws://host:port`.
    public static func connect(to url: URL, identity: String? = "default/swift") async throws -> PieClient {
        try await PieClient(transport: WebSocketTransport(url: url, identity: identity))
    }

    /// Speaks the protocol over `transport`; returns once the server answers a ping.
    public init(transport: some PieTransport) async throws {
        self.transport = transport
        reader = Task { [weak self, frames = transport.frames] in
            do {
                for try await frame in frames {
                    await self?.receive(frame)
                }
                await self?.fail(PieError.connectionClosed)
            } catch {
                await self?.fail(error)
            }
        }
        do {
            try await ping()
        } catch {
            close()
            throw error
        }
    }

    deinit {
        reader?.cancel()
        transport.close()
    }

    public func close() {
        transport.close()
        fail(PieError.connectionClosed)
    }

    public func ping() async throws {
        _ = try await request(.ping)
    }

    /// Launches an installed inferlet (`name` or `name@version`); `input` is
    /// encoded as the JSON its `main` receives.
    public nonisolated func launch(
        _ inferlet: String,
        input: some Encodable,
        captureOutputs: Bool = true
    ) async throws -> PieProcess {
        try await launch(inferlet, json: JSONEncoder().encode(input), captureOutputs: captureOutputs)
    }

    /// Launches an inferlet with `json` as its input, or `{}`.
    public func launch(_ inferlet: String, json: Data? = nil, captureOutputs: Bool = true) async throws -> PieProcess {
        let input = json.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
        let id = try await request(.launch(inferlet: inferlet, input: input, captureOutputs: captureOutputs))
        let (events, continuation) = AsyncThrowingStream.makeStream(of: PieProcess.Event.self)
        continuation.onTermination = { [weak self] termination in
            guard case .cancelled = termination else { return }
            Task { try? await self?.terminate(id) }
        }
        processes[id] = continuation
        for (event, value) in early.removeValue(forKey: id) ?? [] {
            deliver(id, event, value)
        }
        return PieProcess(id: id, events: events, client: self)
    }

    func signal(_ processID: String, _ message: String) async throws {
        if let failure { throw failure }
        try await transport.send(ClientFrame.signal(processID: processID, message: message).encoded(correlationID: 0))
    }

    func terminate(_ processID: String) async throws {
        _ = try await request(.terminate(processID: processID))
    }

    /// Sends `frame` and waits for its response. The slot exists before the
    /// send, so a response racing the send's return is kept, not dropped.
    private func request(_ frame: ClientFrame) async throws -> String {
        if let failure { throw failure }
        nextCorrelationID &+= 1
        let id = nextCorrelationID
        pending[id] = .awaiting(nil)
        do {
            try await transport.send(frame.encoded(correlationID: id))
        } catch {
            pending[id] = nil
            throw error
        }
        return try await withCheckedThrowingContinuation { continuation in
            if case .answered(let result) = pending[id] {
                pending[id] = nil
                continuation.resume(with: result)
            } else {
                pending[id] = .awaiting(continuation)
            }
        }
    }

    private func answer(_ id: UInt32, _ result: Result<String, any Error>) {
        switch pending[id] {
        case .awaiting(let continuation?):
            pending[id] = nil
            continuation.resume(with: result)
        case .awaiting(nil):
            pending[id] = .answered(result)
        case .answered, nil:
            break
        }
    }

    private func receive(_ data: Data) {
        guard let frame = try? ServerFrame(decoding: data) else { return }
        switch frame {
        case .response(let id, let ok, let result):
            answer(id, ok ? .success(result) : .failure(PieError.requestFailed(result)))
        case .processEvent(let id, let event, let value):
            if processes[id] == nil {
                early[id, default: []].append((event, value))
            } else {
                deliver(id, event, value)
            }
        case .other:
            break
        }
    }

    private func deliver(_ id: String, _ event: String, _ value: String) {
        guard let continuation = processes[id] else { return }
        switch event {
        case "stdout": continuation.yield(.stdout(value))
        case "stderr": continuation.yield(.stderr(value))
        case "message": continuation.yield(.message(value))
        case "return":
            continuation.yield(.returned(value))
            continuation.finish()
            processes[id] = nil
        case "error":
            continuation.finish(throwing: PieError.processFailed(value))
            processes[id] = nil
        default:
            break
        }
    }

    private func fail(_ error: any Error) {
        failure = failure ?? error
        for id in pending.keys { answer(id, .failure(error)) }
        for continuation in processes.values { continuation.finish(throwing: error) }
        processes.removeAll()
    }
}

/// A launched inferlet. `events` ends after `.returned`, or throws
/// `PieError.processFailed`; cancelling its iteration terminates the inferlet.
public struct PieProcess: Sendable {
    public enum Event: Sendable, Equatable {
        case stdout(String)
        case stderr(String)
        /// A `session::send` from the inferlet.
        case message(String)
        /// What `main` returned; always the last event.
        case returned(String)
    }

    public let id: String
    public let events: AsyncThrowingStream<Event, any Error>
    let client: PieClient

    /// Delivers `message` to the inferlet's `session::receive`.
    public func send(_ message: String) async throws {
        try await client.signal(id, message)
    }

    public func terminate() async throws {
        try await client.terminate(id)
    }

    /// Drains `events` and returns what `main` returned.
    public func result() async throws -> String {
        for try await case .returned(let value) in events {
            return value
        }
        throw PieError.connectionClosed
    }
}
