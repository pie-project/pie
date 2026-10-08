import Foundation
import PieServerCore
@_exported import PieClient

/// pie in this process: the runtime, the Metal engine and the inferlet
/// sandbox. `connect()` hands out clients whose frames never leave it.
public final class PieServer: @unchecked Sendable {
    public struct Failure: Error, CustomStringConvertible {
        public let description: String
        public init(description: String) { self.description = description }
    }

    /// What booted: model, sku, weight_bytes, kv_pages, kv_page_size, ...
    public let summary: [String: Any]
    let handle: OpaquePointer

    private init(handle: OpaquePointer) {
        self.handle = handle
        let json = String(cString: pie_server_summary(handle))
        summary = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any] ?? [:]
    }

    deinit {
        pie_server_free(handle)
    }

    /// Boots `model` (a `.metal.zt` from `pie model import`). `home` defaults
    /// to `<Caches>/pie`. One server per process.
    public static func start(model: URL, config: Config = Config(), home: URL? = nil) async throws -> PieServer {
        let home = home ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("pie", isDirectory: true)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        let json = try config.json()
        return try await blocking {
            var error: UnsafeMutablePointer<CChar>?
            guard let handle = pie_server_start(model.path, json, home.path, &error) else {
                throw failure(error, "pie_server_start failed")
            }
            return PieServer(handle: handle)
        }
    }

    public func connect() async throws -> PieClient {
        var session: UInt32 = 0
        var error: UnsafeMutablePointer<CChar>?
        guard pie_server_open_session(handle, &session, &error) == 0 else {
            throw Self.failure(error, "open session failed")
        }
        let transport = InProcessTransport(server: self, session: session)
        do {
            return try await PieClient.connect(over: transport)
        } catch {
            transport.close()
            throw error
        }
    }

    /// Installs a program; `file` names it (`x.wasm`, `x.py`, `x.js`).
    /// Returns its `name@version`, which `PieClient.launch` takes.
    public func install(_ bytes: Data, file: String, version: String? = nil) async throws -> String {
        try await Self.blocking { [self] in
            var error: UnsafeMutablePointer<CChar>?
            let name = bytes.withUnsafeBytes { raw in
                pie_server_install(handle, raw.bindMemory(to: UInt8.self).baseAddress, raw.count, file, version, &error)
            }
            guard let name else { throw Self.failure(error, "install failed") }
            defer { pie_string_free(name) }
            return String(cString: name)
        }
    }

    public func install(contentsOf url: URL, version: String? = nil) async throws -> String {
        try await install(Data(contentsOf: url), file: url.lastPathComponent, version: version)
    }

    /// Installs a language component (`python`, `javascript`) for script inferlets.
    public func installLanguage(_ language: String, component: Data) async throws {
        try await Self.blocking { [self] in
            var error: UnsafeMutablePointer<CChar>?
            let status = component.withUnsafeBytes { raw in
                pie_server_install_language(handle, language, raw.bindMemory(to: UInt8.self).baseAddress, raw.count, &error)
            }
            if status != 0 { throw Self.failure(error, "install language failed") }
        }
    }

    /// Stops the runtime and releases the engine's memory; clients fail from
    /// then on. The runtime boots once per process, so no server can start
    /// afterwards.
    public func shutdown() async {
        try? await Self.blocking { [self] in pie_server_shutdown(handle) }
    }

    static func failure(_ error: UnsafeMutablePointer<CChar>?, _ fallback: String) -> Failure {
        guard let error else { return Failure(description: fallback) }
        defer { pie_string_free(error) }
        return Failure(description: String(cString: error))
    }

    private static func blocking<T>(_ body: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            Thread.detachNewThread {
                continuation.resume(with: Result { try body() })
            }
        }
    }
}

extension PieServer {
    /// The boot config (`runtime::embed::BootConfig`), sized for a phone.
    public struct Config: Sendable {
        /// Share of the device's recommended working set the engine may use.
        public var gpuMemoryUtilization: Double = 0.6
        public var maxTotalPages: Int = 512
        public var maxForwardTokens: Int = 512
        public var maxForwardRequests: Int = 4
        public var maxStateSlots: Int = 16
        public var maxModelLength: Int = 4096
        public var sandboxMemoryMB: Int = 256
        public var maxConcurrentProcesses: Int? = 4
        public var sku: String?

        public init() {}

        func json() throws -> String {
            var object: [String: Any] = [
                "gpu_mem_utilization": gpuMemoryUtilization,
                "max_total_pages": maxTotalPages,
                "max_forward_tokens": maxForwardTokens,
                "max_forward_requests": maxForwardRequests,
                "max_state_slots": maxStateSlots,
                "max_model_len": maxModelLength,
                "sandbox_memory_mb": sandboxMemoryMB,
            ]
            if let maxConcurrentProcesses { object["max_concurrent_processes"] = maxConcurrentProcesses }
            if let sku { object["sku"] = sku }
            return String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        }
    }
}

/// One session, drained by a thread of its own. Closing the session wakes
/// that thread at once; the transport keeps the server alive until then.
final class InProcessTransport: PieTransport, @unchecked Sendable {
    private let server: PieServer
    private let session: UInt32
    let frames: AsyncThrowingStream<Data, Error>
    private let continuation: AsyncThrowingStream<Data, Error>.Continuation
    private let lock = NSLock()
    private var closed = false

    init(server: PieServer, session: UInt32) {
        self.server = server
        self.session = session
        (frames, continuation) = AsyncThrowingStream.makeStream()
        Thread.detachNewThread { [self] in pump() }
    }

    func send(_ frame: Data) async throws {
        if lock.withLock({ closed }) { throw PieServer.Failure(description: "the session is closed") }
        var error: UnsafeMutablePointer<CChar>?
        let status = frame.withUnsafeBytes { raw in
            pie_server_send_frame(server.handle, session, raw.bindMemory(to: UInt8.self).baseAddress, raw.count, &error)
        }
        if status != 0 { throw PieServer.failure(error, "send failed") }
    }

    func close() {
        let first = lock.withLock {
            defer { closed = true }
            return !closed
        }
        if first { pie_server_close_session(server.handle, session) }
    }

    private func pump() {
        let sink = Unmanaged.passRetained(Sink(continuation))
        defer {
            sink.release()
            continuation.finish()
        }
        while !lock.withLock({ closed }) {
            var error: UnsafeMutablePointer<CChar>?
            let delivered = pie_server_recv_frames(server.handle, session, 5_000, 64, { ctx, frame, len in
                Unmanaged<Sink>.fromOpaque(ctx!).takeUnretainedValue().continuation.yield(Data(bytes: frame!, count: len))
            }, sink.toOpaque(), &error)
            if delivered < 0 {
                let failure = PieServer.failure(error, "receive failed")
                if !lock.withLock({ closed }) { continuation.finish(throwing: failure) }
                return
            }
        }
    }

    private final class Sink {
        let continuation: AsyncThrowingStream<Data, Error>.Continuation
        init(_ continuation: AsyncThrowingStream<Data, Error>.Continuation) { self.continuation = continuation }
    }
}
