import Foundation

/// Everything version-specific about the Pie side of the app.
///
/// This is the upgrade seam. When a new Pie release changes the engine
/// config schema, the model layout, or the inferlet's name and inputs,
/// this file is the only one that should need editing — `AudioKit`, the
/// conversation controller, and the views know nothing about Pie.
enum PieRuntimeConfig {

    /// Which inferlet serves a conversational turn, and where its wasm
    /// sits inside the app bundle. There is no manifest: the engine
    /// derives the program name from the component when it is installed,
    /// and the version is passed alongside the install.
    struct Inferlet {
        /// Bundle-relative wasm filename, without extension.
        let wasmName: String
        /// Version recorded with the install; must match the crate's.
        let version: String

        var wasmPath: String { "\(Bundle.main.bundlePath)/\(wasmName).wasm" }
    }

    /// The conversational inferlet: one invocation per spoken turn. It is
    /// stateless — the app sends the whole transcript every turn and the
    /// engine serves what it can from the KV state of earlier turns.
    static let voiceChat = Inferlet(wasmName: "voice_chat", version: "0.1.0")

    /// Name under which the engine keeps the conversation's cached state
    /// between turns. One per app: iOS runs a single instance.
    static let sessionName = "ios-voice-session"

    /// A throwaway session for the boot-time warm-up turn, so the state it
    /// leaves behind is never mistaken for the conversation's.
    static let warmUpSessionName = "warmup"

    /// Spoken-reply instructions, sent as the leading system message of
    /// every turn. Lives here rather than in the inferlet so the voice
    /// style can change without rebuilding wasm.
    static let systemPrompt = """
        You are a friendly voice assistant. Everything you say is read \
        aloud by a speech synthesizer, so answer in plain spoken sentences \
        with no lists, headings, code, symbols or markdown. Keep each \
        answer to one to three short sentences unless asked for more.
        """

    /// One rung of the model ladder: a Pie 0.5 model artifact, which is a
    /// directory named as `pie model list` prints it, holding one `.zt`
    /// file named `<slug>.<sku>.<backend>.zt`.
    ///
    /// The app ships whichever rungs are in the bundle; `available`
    /// reports the ones actually present, so a build with only the 0.8B
    /// behaves exactly as before.
    struct Model: Equatable {
        let label: String       // shown in the UI
        let directory: String   // inside models/, e.g. "Qwen--Qwen3.5-0.8B"

        /// What the views and the `-PieModel` launch argument key on.
        /// Kept under its earlier name so the picker needs no change; it
        /// is the artifact directory name now, not a file.
        var fileName: String { directory }

        /// Documents first, then the app bundle. Bundling every rung of the
        /// ladder would mean a multi-gigabyte app; pushing the larger models
        /// into the container keeps the shipped app small and lets a
        /// benchmark run swap models without a rebuild.
        private var candidateDirectories: [String] {
            let documents = FileManager.default
                .urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("models/\(directory)").path
            return [documents, "\(Bundle.main.bundlePath)/models/\(directory)"]
        }

        /// Absolute path of the artifact the engine boots from, or nil
        /// when no rung directory holds one.
        var artifactPath: String? {
            for dir in candidateDirectories {
                if let file = Model.artifact(in: dir) { return "\(dir)/\(file)" }
            }
            return nil
        }

        var isPresent: Bool { artifactPath != nil }

        /// The one `.zt` in an artifact directory. A sharded artifact
        /// stores its shards beside the root as `<stem>-00001.zt`,
        /// `<stem>-00002.zt`, ...; the engine opens the root and finds the
        /// shards itself, so a numbered name is never the one to hand it.
        private static func artifact(in dir: String) -> String? {
            guard let entries = try? FileManager.default.contentsOfDirectory(atPath: dir) else {
                return nil
            }
            let archives = entries.filter { $0.hasSuffix(".zt") }.sorted()
            if archives.count <= 1 { return archives.first }
            let roots = archives.filter { name in
                let stem = String(name.dropLast(".zt".count))
                return stem.range(of: #"-[0-9]+$"#, options: .regularExpression) == nil
            }
            return roots.first ?? archives.first
        }
    }

    /// Ordered smallest first — the ladder used for device benchmarking.
    /// Each rung is imported on the Mac with `pie model import <repo>` and
    /// its directory copied into the bundle or the Documents container.
    static let ladder: [Model] = [
        Model(label: "Qwen3.5-0.8B", directory: "Qwen--Qwen3.5-0.8B"),
        Model(label: "Qwen3.5-2B",   directory: "Qwen--Qwen3.5-2B"),
        Model(label: "Qwen3.5-4B",   directory: "Qwen--Qwen3.5-4B"),
    ]

    static var available: [Model] { ladder.filter(\.isPresent) }

    /// The model the engine boots with. The engine loads weights once per
    /// process, so switching models requires a relaunch — the UI says so
    /// rather than pretending it is live-swappable.
    private(set) static var selected: Model =
        available.first ?? ladder[0]

    /// Persisted across launches so a relaunch comes back on the chosen rung.
    /// The key is launch-argument friendly (no dots): passing
    /// `-PieModel Qwen--Qwen3.5-4B` selects a rung for one run, which is
    /// how the benchmark driver pins each model. The stored value is the
    /// artifact directory name.
    static let modelDefaultsKey = "PieModel"

    static func select(_ model: Model) {
        selected = model
        UserDefaults.standard.set(model.directory, forKey: modelDefaultsKey)
    }

    /// What was requested, whether or not it could be honoured.
    static var requestedModelDirectory: String? {
        UserDefaults.standard.string(forKey: modelDefaultsKey)
    }

    static func restoreSelection() {
        guard let saved = requestedModelDirectory,
              let match = available.first(where: { $0.directory == saved })
        else { return }
        selected = match
    }

    static var modelDescription: String { selected.label }
    static let driverDescription = "Metal"
    static let runtimeDescription = "wasmtime Pulley"

    /// `-PieVerbose 1` (launch argument or `defaults`) turns on the
    /// engine's verbose logging and widens the tracing filter, both into
    /// the mirrored console log.
    private static var verbose: Bool {
        UserDefaults.standard.bool(forKey: "PieVerbose")
    }

    enum Failure: LocalizedError {
        case modelMissing(Model)

        var errorDescription: String? {
            switch self {
            case .modelMissing(let model):
                return "no .zt artifact for \(model.label) under Documents/models/\(model.directory) "
                    + "or the app bundle's models/\(model.directory)"
            }
        }
    }

    /// Points the engine's home directory somewhere iOS actually allows
    /// writes.
    ///
    /// Pie defaults `$PIE_HOME` to `~/.pie`, and on iOS `~` is the app's
    /// container root — which the sandbox makes read-only, so the engine
    /// failed with "Operation not permitted" the first time it created
    /// anything there. Only Documents/, Library/ and tmp/ are writable,
    /// so the home goes to Library/Application Support/pie. The
    /// Simulator's sandbox is permissive enough that this never surfaced
    /// there.
    private static var didPrepareEngineEnvironment = false

    static func prepareEngineEnvironment() throws {
        // Once per process: the engine reads these at boot.
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
        // Everything the engine keeps under its home (weight cache,
        // compiled wasm, installed programs) is regenerable and can grow;
        // keep it out of iCloud/iTunes backups. Nothing there is
        // per-launch: the force-install overwrites the inferlet in place
        // and the caches are keyed by content, so there is nothing to
        // sweep between runs.
        var home = support
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? home.setResourceValues(values)
        setenv("PIE_HOME", support.path, 1)

        // A Rust panic inside the engine aborts the whole process. The
        // message already lands in the mirrored console log; a backtrace
        // there is the difference between a bug report and a shrug.
        setenv("RUST_BACKTRACE", "1", 0)

        // The shim's tracing filter. Two targets are held at WARN even
        // when the rest of the engine is turned up: tarpc logs every RPC
        // at INFO, and the runtime's fire path logs its resolved geometry
        // once per forward pass, which on a decode loop is once per token
        // (a 2-token reply measured 9 such lines). Both would bury the
        // boot and error lines the console mirror exists to capture. A
        // launch environment that sets its own filter wins.
        let quiet = "tarpc=warn,runtime::pipeline::fire=warn"
        setenv("RUST_LOG", verbose ? "debug,\(quiet)" : "info,\(quiet)", 0)
        didPrepareEngineEnvironment = true
    }

    /// Engine config TOML, written to a temp file for the shim to load.
    ///
    /// The same shape `pie config init` writes for a standalone Metal
    /// engine, sized for a phone. The port is 0 so the OS picks a free
    /// loopback port; nothing outside the process ever connects to it.
    static func writeEngineConfig() throws -> String {
        try prepareEngineEnvironment()
        guard let artifact = selected.artifactPath else {
            throw Failure.modelMissing(selected)
        }
        let toml = """
        [server]
        host = "127.0.0.1"
        port = 0
        verbose = \(verbose)

        [model]
        name = "default"
        model = "\(artifact)"

        [engine]
        type = "metal"
        device = ["metal:0"]
        activation_dtype = "bfloat16"
        # Of the device's recommended working set. GPU-touched shared
        # pages are wired on Apple silicon, and the phone still has to
        # run the speech recogniser and synthesizer beside the model.
        gpu_mem_utilization = 0.85
        # 256 pages x 32 tokens = 8k tokens of KV — a long spoken
        # conversation is a few thousand — at a quarter of the default
        # 1024 pages. An iPhone 16 Pro measured a 5.2 GiB ceiling on a
        # single reservation; every GB here counts.
        kv_page_size = 32
        total_pages = 256
        # A voice app serves one short turn at a time; cap the batch so
        # nothing sized by these limits is built for a datacenter.
        max_forward_tokens = 2048
        max_forward_requests = 4
        max_model_len = 8192
        # Hybrid models reserve one recurrent-state slot per seat at boot,
        # and the default 256 would be gigabytes on a phone.
        max_state_slots = 8

        [runtime]
        max_concurrent_processes = 4

        [sandbox]
        # The voice-chat inferlet never touches a filesystem or the
        # network; leaving both off also removes a per-process scratch
        # directory the runtime treats as fatal if it cannot create.
        allow_fs = false
        allow_network = false
        # A phone runs one inferlet at a time. The desktop defaults
        # (1000 instances x 4 GiB) reserve ~4 TB of address space, which
        # iOS refuses.
        max_instances = 4
        max_memory = "128MiB"
        warm_memory = "0B"
        warm_slots = 1
        """
        let path = NSTemporaryDirectory() + "pie-voice-config.toml"
        try toml.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    /// Generation settings for a spoken turn. Short replies matter more
    /// here than in a text UI: every extra token is another second of
    /// someone waiting to be talked at.
    static let maxTokensPerTurn = 120
    static let temperature = 0.7
    static let topP = 0.95
    /// Reasoning text would be spoken aloud as a stream of consciousness
    /// before the answer; keep the model in its direct-answer mode.
    static let think = false
}
