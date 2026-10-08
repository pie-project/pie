import Foundation

// The C ABI exported by libpie_ios_shim.a (ios/pie-shim). Declared with
// @_silgen_name so the app needs no bridging header or module map — the
// staticlib is linked directly by build-app.sh.
//
// Nothing above this file should reference these symbols.

@_silgen_name("pie_ios_run_stream")
private func pie_ios_run_stream(
    _ configPath: UnsafePointer<CChar>,
    _ wasmPath: UnsafePointer<CChar>,
    _ manifestPath: UnsafePointer<CChar>,
    _ inferletId: UnsafePointer<CChar>,
    _ inputJson: UnsafePointer<CChar>,
    _ cb: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void,
    _ ctx: UnsafeMutableRawPointer?
) -> UnsafeMutablePointer<CChar>?

@_silgen_name("pie_ios_free")
private func pie_ios_free(_ s: UnsafeMutablePointer<CChar>?)

/// Carries the per-chunk closure across the C boundary. The shim invokes
/// the callback on the calling thread, so no locking is needed here.
private final class DeltaSink {
    let onDelta: (String) -> Void
    init(_ onDelta: @escaping (String) -> Void) { self.onDelta = onDelta }
}

enum PieBridge {
    /// Runs one inferlet to completion, synchronously, on the current
    /// thread. `onDelta` fires for each stdout chunk the inferlet emits.
    /// Returns the inferlet's return value.
    ///
    /// Blocking is deliberate: the shim owns a tokio runtime and the
    /// engine is process-global, so the caller decides the concurrency
    /// policy. `PieEngine` runs this on a serial background queue.
    static func runStreaming(
        configPath: String,
        wasmPath: String,
        manifestPath: String,
        inferletId: String,
        inputJSON: String,
        onDelta: @escaping (String) -> Void
    ) -> String {
        let sink = DeltaSink(onDelta)
        let ctx = Unmanaged.passRetained(sink).toOpaque()
        defer { Unmanaged<DeltaSink>.fromOpaque(ctx).release() }

        let trampoline: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = {
            chunk, ctx in
            guard let chunk, let ctx else { return }
            let sink = Unmanaged<DeltaSink>.fromOpaque(ctx).takeUnretainedValue()
            sink.onDelta(String(cString: chunk))
        }

        guard let raw = pie_ios_run_stream(
            configPath, wasmPath, manifestPath, inferletId, inputJSON, trampoline, ctx
        ) else {
            return "PIE ERROR: shim returned null"
        }
        defer { pie_ios_free(raw) }
        return String(cString: raw)
    }
}
