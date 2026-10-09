import Foundation

// The C ABI exported by libpie_ios_shim.a (ios/pie-shim), version 3.
// Declared with @_silgen_name so the app needs no bridging header or
// module map: the staticlib is linked straight from the repository target
// directory by project.yml.
//
// Nothing above this file should reference these symbols.

@_silgen_name("pie_ios_run_stream")
private func pie_ios_run_stream(
    _ configPath: UnsafePointer<CChar>,
    _ wasmPath: UnsafePointer<CChar>,
    _ version: UnsafePointer<CChar>?,
    _ inputJson: UnsafePointer<CChar>,
    _ turnId: UInt64,
    _ cb: @convention(c) (Int32, UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void,
    _ ctx: UnsafeMutableRawPointer?
) -> UnsafeMutablePointer<CChar>?

@_silgen_name("pie_ios_cancel")
private func pie_ios_cancel(_ turnId: UInt64)

@_silgen_name("pie_ios_free")
private func pie_ios_free(_ s: UnsafeMutablePointer<CChar>?)

enum PieBridge {

    /// Which channel of the inferlet a streamed chunk came from.
    enum Channel {
        /// The inferlet's stdout: reply text.
        case reply
        /// The inferlet's session messages: reasoning, in thinking mode.
        case reasoning
    }

    /// How a run ended, decoded from the shim's return string.
    enum Outcome {
        /// The inferlet's return value.
        case completed(String)
        /// The whole "PIE ERROR: ..." message.
        case failed(String)
        /// `cancel(turnID:)` stopped the run, before or during it. Every
        /// chunk produced before the stop has already been delivered.
        case cancelled
    }

    /// Runs one inferlet to completion, synchronously, on the current
    /// thread. `onChunk` fires on this same thread for every chunk the
    /// inferlet streams, in order.
    ///
    /// The first call boots the engine from the config at `configPath`
    /// and keeps it warm for the life of the process; the wasm at
    /// `wasmPath` is installed once per process on its first use, so
    /// passing the same paths every turn costs nothing after the first.
    ///
    /// Blocking is deliberate: the shim owns a tokio runtime and the
    /// engine is process-global, so the caller decides the concurrency
    /// policy. `PieEngine` runs this on a serial background queue.
    static func run(
        configPath: String,
        wasmPath: String,
        version: String?,
        inputJSON: String,
        turnID: UInt64,
        onChunk: @escaping (Channel, String) -> Void
    ) -> Outcome {
        let sink = ChunkSink(onChunk)
        let ctx = Unmanaged.passRetained(sink).toOpaque()
        defer { Unmanaged<ChunkSink>.fromOpaque(ctx).release() }

        let trampoline: @convention(c) (Int32, UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = {
            kind, chunk, ctx in
            guard let chunk, let ctx else { return }
            let sink = Unmanaged<ChunkSink>.fromOpaque(ctx).takeUnretainedValue()
            // Kind 1 is the session-message channel. Anything else is
            // treated as reply text, so a shim that grows a new kind
            // degrades to showing it rather than dropping it.
            sink.onChunk(kind == 1 ? .reasoning : .reply, String(cString: chunk))
        }

        // The version is the one nullable argument: Swift bridges a
        // String to a C string implicitly, but not an Optional one, so
        // the two cases are spelled out.
        let raw: UnsafeMutablePointer<CChar>?
        if let version {
            raw = pie_ios_run_stream(configPath, wasmPath, version, inputJSON, turnID, trampoline, ctx)
        } else {
            raw = pie_ios_run_stream(configPath, wasmPath, nil, inputJSON, turnID, trampoline, ctx)
        }
        guard let raw else {
            return .failed("PIE ERROR: shim returned null")
        }
        defer { pie_ios_free(raw) }
        let result = String(cString: raw)

        if result == "PIE CANCELLED" { return .cancelled }
        if result.hasPrefix("PIE ERROR") { return .failed(result) }
        return .completed(result)
    }

    /// Stops the run that was given `turnID`: terminates its engine
    /// process if it is running, makes it return `.cancelled` without
    /// launching if it has not started, and does nothing if it has
    /// finished. Thread-safe and non-blocking, so it is called directly
    /// from the main thread while another thread is inside `run`.
    static func cancel(turnID: UInt64) {
        pie_ios_cancel(turnID)
    }
}

/// Carries the per-chunk closure across the C boundary. The shim invokes
/// the callback on the thread that called `run`, so no locking is needed.
private final class ChunkSink {
    let onChunk: (PieBridge.Channel, String) -> Void
    init(_ onChunk: @escaping (PieBridge.Channel, String) -> Void) { self.onChunk = onChunk }
}
