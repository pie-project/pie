import Foundation

// The C ABI exported by libpie_ios_shim.a (ios/pie-shim). Declared with
// @_silgen_name so the app needs no bridging header or module map — the
// staticlib is linked straight from the repository target directory by
// project.yml.
//
// Nothing above this file should reference these symbols.

@_silgen_name("pie_ios_run_stream")
private func pie_ios_run_stream(
    _ configPath: UnsafePointer<CChar>,
    _ wasmPath: UnsafePointer<CChar>,
    _ version: UnsafePointer<CChar>?,
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
    /// The first call boots the engine from the config at `configPath`
    /// and keeps it warm for the life of the process; the wasm at
    /// `wasmPath` is installed once per process on its first use, so
    /// passing the same paths every turn costs nothing after the first.
    ///
    /// Blocking is deliberate: the shim owns a tokio runtime and the
    /// engine is process-global, so the caller decides the concurrency
    /// policy. `PieEngine` runs this on a serial background queue.
    static func runStreaming(
        configPath: String,
        wasmPath: String,
        version: String?,
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

        // The version is the one nullable argument: Swift bridges a
        // String to a C string implicitly, but not an Optional one, so
        // the two cases are spelled out.
        let raw: UnsafeMutablePointer<CChar>?
        if let version {
            raw = pie_ios_run_stream(configPath, wasmPath, version, inputJSON, trampoline, ctx)
        } else {
            raw = pie_ios_run_stream(configPath, wasmPath, nil, inputJSON, trampoline, ctx)
        }
        guard let raw else {
            return "PIE ERROR: shim returned null"
        }
        defer { pie_ios_free(raw) }
        return String(cString: raw)
    }
}
