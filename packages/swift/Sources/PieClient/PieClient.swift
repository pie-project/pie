import Foundation

/// A client of a pie server: a remote `pie serve`, or a `PieServer` in this
/// process.
public actor PieClient {
    private let transport: any PieTransport
    private var nextCorrelationID: UInt32 = 0
    private var lastHandle: UInt64 = 0
    private var pending: [UInt32: Slot] = [:]
    private var processes: [String: AsyncThrowingStream<PieProcess.Event, any Error>.Continuation] = [:]
    /// Files part-way through their chunks, by process and hash: two reads of
    /// one handle share a hash.
    private var downloads: [String: [String: Download]] = [:]
    /// Frames that arrived before their process's launch response.
    private var early: [String: [ServerFrame]] = [:]
    private var failure: (any Error)?
    private var reader: Task<Void, Never>?

    private enum Slot {
        case awaiting(CheckedContinuation<String, any Error>?)
        case answered(Result<String, any Error>)
    }

    private struct Download {
        let name: String?
        var data = Data()
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
        captureOutputs: Bool = true,
        handles: [String: any Handle] = [:],
        returns: [String: any Handle] = [:]
    ) async throws -> PieProcess {
        try await launch(
            inferlet, json: JSONEncoder().encode(input), captureOutputs: captureOutputs,
            handles: handles, returns: returns
        )
    }

    /// Launches an inferlet with `json` as its input, or `{}`. `handles` move
    /// into the process, by the name its guest takes each under, and cannot
    /// be used until they come back: fork first to keep one. `returns` is
    /// what the process gives back, by the name it gives each under.
    public func launch(
        _ inferlet: String,
        json: Data? = nil,
        captureOutputs: Bool = true,
        handles: [String: any Handle] = [:],
        returns: [String: any Handle] = [:]
    ) async throws -> PieProcess {
        let input = json.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
        let frame = ClientFrame.launch(
            inferlet: inferlet, input: input, captureOutputs: captureOutputs,
            handles: handles.mapValues { $0.name }, returns: returns.mapValues { $0.name }
        )
        let id: String
        do {
            id = try await request(frame)
        } catch PieError.requestFailed(let reason) {
            if let code = HandleErrorCode(refusal: reason) { throw PieError.handleFailed(code, reason) }
            throw PieError.requestFailed(reason)
        }
        let (events, continuation) = AsyncThrowingStream.makeStream(of: PieProcess.Event.self)
        continuation.onTermination = { [weak self] termination in
            guard case .cancelled = termination else { return }
            Task { try? await self?.terminate(id) }
        }
        processes[id] = continuation
        for frame in early.removeValue(forKey: id) ?? [] {
            route(frame)
        }
        return PieProcess(id: id, events: events, client: self)
    }

    /// A new, empty working set.
    public func workingSet() async throws -> WorkingSet {
        let name = nextHandle()
        _ = try await handleOp(.new(name: name, kind: .workingSet))
        return WorkingSet(client: self, name: name)
    }

    /// A matcher at the start of one grammar, or of any JSON given none. The
    /// grammar compiles where a process binds it, and an error in it fails
    /// that launch.
    public func matcher(jsonSchema: String? = nil, regex: String? = nil, ebnf: String? = nil) async throws -> Matcher {
        let grammar: HandleOp.Grammar =
            if let jsonSchema { .jsonSchema(jsonSchema) }
            else if let regex { .regex(regex) }
            else if let ebnf { .ebnf(ebnf) }
            else { .json }
        let name = nextHandle()
        _ = try await handleOp(.new(name: name, kind: .matcher(grammar)))
        return Matcher(client: self, name: name)
    }

    /// An upload, if `data` is given: encoded pictures, which decode where a
    /// process binds them. Else a name for a process to give pictures to.
    public func frames(_ data: Data? = nil) async throws -> Frames {
        let name = nextHandle()
        if let data { _ = try await handleOp(.new(name: name, kind: .frames(data))) }
        return Frames(client: self, name: name)
    }

    /// A name for a process to give sound to.
    public func pcm() -> Pcm {
        Pcm(client: self, name: nextHandle())
    }

    /// A new working set as `snapshot` stands.
    public func loadSnapshot(_ snapshot: String) async throws -> WorkingSet {
        let name = nextHandle()
        _ = try await handleOp(.load(name: name, snapshot: snapshot))
        return WorkingSet(client: self, name: name)
    }

    /// Handles loaded from it stay.
    public func deleteSnapshot(_ snapshot: String) async throws {
        _ = try await handleOp(.deleteSnapshot(snapshot: snapshot))
    }

    /// Each snapshot under `prefix`.
    public func listSnapshots(prefix: String = "") async throws -> [Snapshot] {
        struct Listing: Decodable { let snapshots: [Snapshot] }
        return try decoded(Listing.self, from: await handleOp(.listSnapshots(prefix: prefix))).snapshots
    }

    func save(_ name: UInt64, snapshot: String) async throws -> Snapshot {
        struct Kept: Decodable { let tokens: Int; let bytes: Int }
        let kept = try decoded(Kept.self, from: await handleOp(.save(name: name, snapshot: snapshot)))
        return Snapshot(name: snapshot, tokens: kept.tokens, bytes: kept.bytes)
    }

    private func decoded<T: Decodable>(_ type: T.Type, from answer: Data) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: answer)
        } catch {
            throw PieError.malformedFrame("a handle answer: \(error)")
        }
    }

    func nextHandle() -> UInt64 {
        lastHandle += 1
        return lastHandle
    }

    /// Sends `op` alone and returns its answer as JSON.
    func handleOp(_ op: HandleOp) async throws -> Data {
        let result = try await handleRequest(.handles(op))
        guard let answers = try? JSONSerialization.jsonObject(with: Data(result.utf8)) as? [[String: Any]],
              let answer = answers.first else {
            throw PieError.malformedFrame("a handle answer that is not a list of objects")
        }
        if let error = answer["error"] {
            let reason = error as? String ?? String(describing: error)
            let code = (answer["code"] as? String).flatMap { HandleErrorCode(rawValue: $0) }
            throw PieError.handleFailed(code ?? HandleErrorCode(refusal: reason), reason)
        }
        return try JSONSerialization.data(withJSONObject: answer)
    }

    private func handleRequest(_ frame: ClientFrame) async throws -> String {
        do {
            return try await request(frame)
        } catch PieError.requestFailed(let reason) {
            throw PieError.handleFailed(HandleErrorCode(refusal: reason), reason)
        }
    }

    func read(_ name: UInt64, format: String) async throws -> Data {
        if let failure { throw failure }
        // Picked here so the stream is open before the first chunk can arrive;
        // lowercased because the server's events carry its own form of the id.
        let reader = UUID().uuidString.lowercased()
        let (events, continuation) = AsyncThrowingStream.makeStream(of: PieProcess.Event.self)
        processes[reader] = continuation
        defer {
            processes[reader] = nil
            downloads[reader] = nil
        }
        do {
            _ = try await handleRequest(.readHandle(name: name, format: format, reader: reader))
            var data = Data()
            for try await event in events {
                switch event {
                case .file(_, let bytes): data = bytes
                case .returned: return data
                default: break
                }
            }
        } catch PieError.processFailed(let reason) {
            throw PieError.handleFailed(HandleErrorCode(refusal: reason), reason)
        }
        // Only a cancelled task leaves the stream short of `.returned`.
        try Task.checkCancellation()
        throw PieError.connectionClosed
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
        case .processEvent(let id, _, _), .file(let id, _, _, _, _, _):
            if processes[id] == nil {
                early[id, default: []].append(frame)
            } else {
                route(frame)
            }
        case .other:
            break
        }
    }

    private func route(_ frame: ServerFrame) {
        switch frame {
        case .processEvent(let id, let event, let value):
            deliver(id, event, value)
        case .file(let id, let hash, let index, let total, let data, let name):
            collect(id, hash, index, total, data, name)
        case .response, .other:
            break
        }
    }

    private func collect(_ id: String, _ hash: String, _ index: Int, _ total: Int, _ data: Data, _ name: String?) {
        guard let continuation = processes[id], index == 0 || downloads[id]?[hash] != nil else { return }
        downloads[id, default: [:]][hash, default: Download(name: name)].data.append(data)
        if index == total - 1, let download = downloads[id]?.removeValue(forKey: hash) {
            continuation.yield(.file(name: download.name, data: download.data))
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
            downloads[id] = nil
        case "error":
            continuation.finish(throwing: PieError.processFailed(value))
            processes[id] = nil
            downloads[id] = nil
        default:
            break
        }
    }

    private func fail(_ error: any Error) {
        failure = failure ?? error
        for id in pending.keys { answer(id, .failure(error)) }
        for continuation in processes.values { continuation.finish(throwing: error) }
        processes.removeAll()
        downloads.removeAll()
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
        /// A file the inferlet sent (`session::send_file_as`, `send_frames`),
        /// under the name it gave, if any.
        case file(name: String?, data: Data)
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
