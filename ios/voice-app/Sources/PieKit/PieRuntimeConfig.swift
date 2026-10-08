import Foundation

/// Everything version-specific about the Pie side of the app.
///
/// This is the upgrade seam. When a new Pie release changes the engine
/// config schema, the model layout, or an inferlet's id/manifest, this
/// file is the only one that should need editing — `AudioKit`, the
/// conversation controller, and the views know nothing about Pie.
enum PieRuntimeConfig {

    /// Which inferlet serves a conversational turn, and where its
    /// artifacts sit inside the app bundle.
    struct Inferlet {
        /// `name@version`, matching the inferlet's Pie.toml.
        let id: String
        /// Bundle-relative wasm filename, without extension.
        let wasmName: String
        /// Bundle-relative manifest filename, without extension.
        let manifestName: String

        var wasmPath: String { "\(Bundle.main.bundlePath)/\(wasmName).wasm" }
        var manifestPath: String { "\(Bundle.main.bundlePath)/\(manifestName).toml" }
    }

    /// The conversational inferlet: one invocation per spoken turn,
    /// carrying KV state across turns in a named snapshot.
    static let voiceChat = Inferlet(
        id: "voice-chat@0.1.0",
        wasmName: "voice_chat",
        manifestName: "voice-chat-Pie"
    )

    /// KV snapshot name the conversation lives under, inside the engine.
    static let sessionName = "ios-voice-session"

    /// Model + driver, as the engine reports them. Shown in the UI so a
    /// screenshot says what actually ran.
    /// One rung of the model ladder. The app ships whichever GGUFs are in
    /// the bundle; `available` reports the ones actually present, so a
    /// build with only the 0.6B behaves exactly as before.
    struct Model: Equatable {
        let label: String       // shown in the UI
        let fileName: String    // inside qwen3-gguf/
        /// Documents first, then the app bundle. Bundling every rung of the
        /// ladder would mean a multi-gigabyte app; pushing the larger models
        /// into the container keeps the shipped app small and lets a
        /// benchmark run swap models without a rebuild.
        var path: String {
            let documents = FileManager.default
                .urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("qwen3-gguf/\(fileName)").path
            if FileManager.default.fileExists(atPath: documents) { return documents }
            return "\(Bundle.main.bundlePath)/qwen3-gguf/\(fileName)"
        }
        var isPresent: Bool { FileManager.default.fileExists(atPath: path) }
    }

    /// Ordered smallest first — the ladder used for device benchmarking.
    static let ladder: [Model] = [
        Model(label: "Qwen3-0.6B Q4_K_M", fileName: "Qwen3-0.6B-Q4_K_M.gguf"),
        Model(label: "Qwen3-1.7B Q4_K_M", fileName: "Qwen3-1.7B-Q4_K_M.gguf"),
        Model(label: "Qwen3-4B Q4_K_M",   fileName: "Qwen3-4B-Q4_K_M.gguf"),
        Model(label: "Qwen3-8B Q4_K_M",   fileName: "Qwen3-8B-Q4_K_M.gguf"),
    ]

    static var available: [Model] { ladder.filter(\.isPresent) }

    /// The model the engine boots with. The engine loads weights once per
    /// process, so switching models requires a relaunch — the UI says so
    /// rather than pretending it is live-swappable.
    private(set) static var selected: Model =
        available.first ?? ladder[0]

    /// Persisted across launches so a relaunch comes back on the chosen rung.
    /// The key is launch-argument friendly (no dots): passing
    /// `-PieModel Qwen3-4B-Q4_K_M.gguf` selects a rung for one run, which is
    /// how the benchmark driver pins each model.
    static let modelDefaultsKey = "PieModel"

    static func select(_ model: Model) {
        selected = model
        UserDefaults.standard.set(model.fileName, forKey: modelDefaultsKey)
    }

    /// What was requested, whether or not it could be honoured.
    static var requestedModelFile: String? {
        UserDefaults.standard.string(forKey: modelDefaultsKey)
    }

    static func restoreSelection() {
        guard let saved = requestedModelFile,
              let match = available.first(where: { $0.fileName == saved })
        else { return }
        selected = match
    }

    static var modelDescription: String { selected.label }
    static let driverDescription = "ggml (CPU)"
    static let runtimeDescription = "wasmtime Pulley"

    /// Engine config TOML, written to a temp file for the shim to load.
    ///
    /// Mirrors what `pie run` would read from disk. The port is
    /// irrelevant — the shim rewrites it to 0 and lets the OS pick — but
    /// the config schema requires the section.
    /// Points the engine's state directory somewhere iOS actually allows
    /// writes.
    ///
    /// The server defaults `$PIE_HOME` to `~/.pie`, and on iOS `~` is the
    /// app's container root — which the sandbox makes read-only, so the
    /// driver failed with "create state dir ... Operation not permitted".
    /// Only Documents/, Library/ and tmp/ are writable, so state goes to
    /// Library/Application Support/pie. The Simulator's sandbox is
    /// permissive enough that this never surfaced there.
    private static var didPrepareEngineEnvironment = false

    static func prepareEngineEnvironment() throws {
        // Once per process: the engine reads these at boot, and the
        // cleanup below must never run while an engine is alive.
        guard !didPrepareEngineEnvironment else { return }

        let support = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("pie", isDirectory: true)

        try FileManager.default.createDirectory(
            at: support, withIntermediateDirectories: true
        )
        // Engine scratch (logs, program cache, per-launch state) is
        // regenerable and can grow; keep it out of iCloud/iTunes backups.
        var stateRoot = support
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? stateRoot.setResourceValues(values)
        setenv("PIE_HOME", support.path, 1)

        // Each launch gets its own `standalone/<pid>/` state directory and
        // nothing removes it afterwards. iOS never runs two instances of
        // the app, so anything there belongs to a dead process — clear it
        // before boot rather than letting a crash-loop pile them up.
        let stale = support.appendingPathComponent("standalone", isDirectory: true)
        try? FileManager.default.removeItem(at: stale)

        // A Rust panic inside the engine aborts the whole process. The
        // message already lands in the mirrored console log; a backtrace
        // there is the difference between a bug report and a shrug.
        setenv("RUST_BACKTRACE", "1", 0)

        // ggml defaults to every hardware thread. A phone's efficiency
        // cores finish each fork/join phase last, so all-cores is slower
        // than leaving a couple free (llama.cpp's own iOS default is
        // cores - 2). Respect an override from a launch environment.
        if getenv("GGML_N_THREADS") == nil {
            let cores = ProcessInfo.processInfo.activeProcessorCount
            let threads = max(2, min(8, cores - 2))
            setenv("GGML_N_THREADS", String(threads), 1)
        }
        didPrepareEngineEnvironment = true
    }

    static func writeEngineConfig() throws -> String {
        try prepareEngineEnvironment()
        // `-PieVerbose 1` (launch argument or `defaults`) turns on the
        // engine's and driver's verbose logging — graph node counts, KV
        // allocation, thread pinning — into the mirrored console log.
        let verbose = UserDefaults.standard.bool(forKey: "PieVerbose")
        let toml = """
        [server]
        host = "127.0.0.1"
        port = 8093
        verbose = \(verbose)

        [auth]
        enabled = false

        [runtime]
        # The voice-chat inferlet never touches a filesystem; leaving the
        # scratch mount off also removes a per-turn directory creation
        # that the runtime treats as fatal if it fails.
        allow_fs = false
        allow_network = false
        # A phone runs one inferlet at a time. The desktop defaults
        # (1000 instances x 4 GiB) reserve ~4 TB of address space, which
        # iOS refuses; the engine clamps these further on iOS regardless.
        wasm_max_instances = 4
        wasm_max_memory_mb = 128

        [[model]]
        name = "default"
        hf_repo = "\(selected.path)"

        [model.driver]
        type = "portable"
        device = ["cpu"]

        [model.driver.options]
        # 256 pages x 32 tokens = 8k tokens of KV — a long spoken
        # conversation is a few thousand — at a quarter of the default
        # 1024 pages' address space. An iPhone 16 Pro measured a 5.2 GiB
        # ceiling on a single reservation; every GB here counts.
        total_pages = 256
        # A voice app serves one short turn at a time; cap the batch so
        # nothing sized by these limits is built for a datacenter.
        max_forward_tokens = 2048
        max_forward_requests = 8
        # Weight load from flash on a phone, possibly while iOS suspends
        # the app in the background: give the driver far longer than the
        # desktop default (120 s) before boot is declared failed.
        ready_timeout_s = 600
        """
        let path = NSTemporaryDirectory() + "pie-voice-config.toml"
        try toml.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    /// Generation settings for a spoken turn. Short replies matter more
    /// here than in a text UI: every extra token is another second of
    /// someone waiting to be talked at.
    static let maxTokensPerTurn = 110
    static let temperature = 0.7
    static let topP = 0.95
}
