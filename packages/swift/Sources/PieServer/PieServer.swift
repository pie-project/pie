import Foundation
@_exported import PieClient
import PieServerCore
import Synchronization

/// pie in this process: the runtime, the Metal engine and the inferlet
/// sandbox. `connect()` hands out clients whose frames never leave it.
public final class PieServer: Sendable {
    /// What booted.
    public struct Summary: Decodable, Sendable, Equatable {
        public let model: String
        public let deployment: String
        public let trace: String
        public let weightBytes: Int
        public let kvPages: Int
        public let kvPageSize: Int
        public let maxLanes: Int
        public let maxTokens: Int
    }

    /// The boot configuration (`runtime::embed::BootConfig`), sized for a phone.
    public struct Configuration: Encodable, Sendable, Equatable {
        /// Share of the device's recommended working set the engine may use.
        public var gpuMemoryUtilization = 0.6
        public var maxTotalPages = 512
        public var maxForwardTokens = 512
        public var maxForwardRequests = 4
        public var maxStateSlots = 16
        public var maxModelLength = 4096
        /// The cap on each inferlet's linear memory.
        public var sandboxMemoryMB = 256
        public var maxConcurrentProcesses: Int? = 4

        public init() {}

        enum CodingKeys: String, CodingKey {
            case gpuMemoryUtilization = "gpu_mem_utilization"
            case maxTotalPages = "max_total_pages"
            case maxForwardTokens = "max_forward_tokens"
            case maxForwardRequests = "max_forward_requests"
            case maxStateSlots = "max_state_slots"
            case maxModelLength = "max_model_len"
            case sandboxMemoryMB = "sandbox_memory_mb"
            case maxConcurrentProcesses = "max_concurrent_processes"
        }
    }

    public enum Language: String, Sendable {
        case python, javascript
    }

    public let summary: Summary
    let handle: Handle

    /// The C handle; every call on it is safe from any thread.
    struct Handle: @unchecked Sendable {
        let pointer: OpaquePointer
    }

    private init(handle: Handle) throws {
        self.handle = handle
        summary = try JSONDecoder.snakeCase.decode(Summary.self, from: Data(String(cString: pie_server_summary(handle.pointer)).utf8))
    }

    deinit {
        let handle = handle
        Thread.detachNewThread { pie_server_free(handle.pointer) }
    }

    /// Boots `model` (a `.metal.zt` from `pie model import`); `home` holds the
    /// inferlet cache and defaults to `<Caches>/pie`. One server per process.
    public static func start(
        model: URL,
        configuration: Configuration = Configuration(),
        home: URL = .cachesDirectory.appending(path: "pie", directoryHint: .isDirectory)
    ) async throws -> PieServer {
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        let config = try String(decoding: JSONEncoder().encode(configuration), as: UTF8.self)
        let handle = try await blocking {
            Handle(pointer: try call { error in pie_server_start(model.path(), config, home.path(), error) })
        }
        do {
            return try PieServer(handle: handle)
        } catch {
            pie_server_free(handle.pointer)
            throw error
        }
    }

    public func connect() async throws -> PieClient {
        var session: UInt32 = 0
        try Self.check { error in pie_server_open_session(handle.pointer, &session, error) }
        let transport = InProcessTransport(server: self, session: session)
        do {
            return try await PieClient(transport: transport)
        } catch {
            transport.close()
            throw error
        }
    }

    /// Installs a program; `file` names it (`x.wasm`, `x.py`, `x.js`).
    /// Returns its `name@version`, which `PieClient.launch` takes.
    public func install(_ program: Data, file: String, version: String? = nil) async throws -> String {
        try await Self.blocking { [handle] in
            let name = try Self.call { error in
                program.withUnsafeBytes { bytes in
                    pie_server_install(handle.pointer, bytes.baseAddress, bytes.count, file, version, error)
                }
            }
            defer { pie_string_free(name) }
            return String(cString: name)
        }
    }

    public func install(contentsOf url: URL, version: String? = nil) async throws -> String {
        try await install(Data(contentsOf: url), file: url.lastPathComponent, version: version)
    }

    /// Installs the component that runs script inferlets in `language`.
    public func installLanguage(_ language: Language, component: Data) async throws {
        try await Self.blocking { [handle] in
            try Self.check { error in
                component.withUnsafeBytes { bytes in
                    pie_server_install_language(handle.pointer, language.rawValue, bytes.baseAddress, bytes.count, error)
                }
            }
        }
    }

    /// Stops the runtime and releases the engine's memory; clients fail from
    /// then on. The runtime boots once per process, so no server can start
    /// afterwards.
    public func shutdown() async {
        try? await Self.blocking { [handle] in pie_server_shutdown(handle.pointer) }
    }

    /// Runs a C call that returns NULL on failure.
    static func call<T>(_ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> T?) throws -> T {
        var error: UnsafeMutablePointer<CChar>?
        guard let value = body(&error) else { throw failure(error) }
        return value
    }

    /// Runs a C call that returns nonzero on failure.
    static func check(_ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32) throws {
        var error: UnsafeMutablePointer<CChar>?
        if body(&error) != 0 { throw failure(error) }
    }

    static func failure(_ error: UnsafeMutablePointer<CChar>?) -> PieError {
        guard let error else { return .server("pie failed without a message") }
        defer { pie_string_free(error) }
        return .server(String(cString: error))
    }

    /// The C calls block for as long as a boot or a compile takes, so each
    /// runs on a thread of its own rather than the cooperative pool.
    private static func blocking<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            Thread.detachNewThread {
                continuation.resume(with: Result { try body() })
            }
        }
    }
}

extension JSONDecoder {
    static var snakeCase: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }
}

/// One session, drained by a thread of its own. Closing the session wakes
/// that thread at once; the transport keeps the server alive until then.
final class InProcessTransport: PieTransport {
    let frames: AsyncThrowingStream<Data, any Error>
    private let server: PieServer
    private let session: UInt32
    private let continuation: AsyncThrowingStream<Data, any Error>.Continuation
    private let closed = Mutex(false)

    init(server: PieServer, session: UInt32) {
        self.server = server
        self.session = session
        (frames, continuation) = AsyncThrowingStream.makeStream(of: Data.self)
        Thread.detachNewThread { [self] in pump() }
    }

    func send(_ frame: Data) async throws {
        if closed.withLock({ $0 }) { throw PieError.connectionClosed }
        try PieServer.check { error in
            frame.withUnsafeBytes { bytes in
                pie_server_send_frame(server.handle.pointer, session, bytes.baseAddress, bytes.count, error)
            }
        }
    }

    func close() {
        let first = closed.withLock { closed in
            defer { closed = true }
            return !closed
        }
        if first { pie_server_close_session(server.handle.pointer, session) }
    }

    private func pump() {
        let sink = Unmanaged.passRetained(Sink(continuation))
        defer {
            sink.release()
            continuation.finish()
        }
        while !closed.withLock({ $0 }) {
            var error: UnsafeMutablePointer<CChar>?
            let delivered = pie_server_recv_frames(server.handle.pointer, session, 5_000, 64, { context, frame, length in
                Unmanaged<Sink>.fromOpaque(context!).takeUnretainedValue().continuation.yield(Data(bytes: frame!, count: length))
            }, sink.toOpaque(), &error)
            if delivered < 0 {
                let failure = PieServer.failure(error)
                if !closed.withLock({ $0 }) { continuation.finish(throwing: failure) }
                return
            }
        }
    }

    private final class Sink: Sendable {
        let continuation: AsyncThrowingStream<Data, any Error>.Continuation
        init(_ continuation: AsyncThrowingStream<Data, any Error>.Continuation) { self.continuation = continuation }
    }
}
