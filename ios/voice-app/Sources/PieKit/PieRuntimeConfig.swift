import Foundation

/// Everything version-specific about the Pie side of the app.
///
/// This is the upgrade seam. When a new Pie release changes the engine
/// config schema, the model layout, or the inferlet's name and inputs,
/// this file is the only one that should need editing. The controllers
/// take their prompts and generation presets from here, and the views
/// its descriptions; neither knows anything else about Pie, and
/// `AudioKit` knows nothing at all.
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

    /// The conversational inferlet: one invocation per reply, typed or
    /// spoken. It is stateless: the app sends the whole transcript every
    /// turn and the engine serves what it can from the KV state that
    /// earlier turns of the same session published.
    static let voiceChat = Inferlet(wasmName: "voice_chat", version: "0.1.0")

    /// A throwaway session for the boot-time warm-up turn, so the state it
    /// leaves behind is never mistaken for a conversation's.
    static let warmUpSessionName = "warmup"

    /// The session title replies run in. Titles share one namespace so
    /// their short fixed prompt is served from cache after the first.
    static let titleSessionName = "titles"

    /// Tokens the engine holds for one sequence: prompt plus reply. Sizes
    /// `max_model_len` in the engine config and the prompt budget the
    /// engine trims the transcript to.
    static let contextTokens = 8192

    /// The most of one message's attachments the model is shown, in
    /// estimated tokens, shared among them, and the most any one of them
    /// gets. Half the context at most, so a message with attachments still
    /// leaves room for the reply (Thinking's 1,536 tokens), the system
    /// prompt and some of the conversation before it.
    static let attachmentTokensPerMessage = 4096
    static let attachmentTokensEach = 2048

    // MARK: - Prompts

    /// Instructions for a typed chat, sent as the leading system message.
    /// Lives here rather than in the inferlet so the style can change
    /// without rebuilding wasm.
    static let chatSystemPrompt = """
        You are Pie, a helpful assistant that runs entirely on this iPhone, \
        with no server involved. You may use Markdown \
        (lists, bold, tables, code blocks) when it makes an answer clearer. \
        Be concise by default and go into detail only when asked. You are a \
        small model, so when you do not know something, say "I'm not sure" \
        rather than inventing an answer.
        """

    /// Added to every question asked in voice mode, where the model reads
    /// it: the reply is read aloud, so no formatting survives, and a small
    /// model follows an instruction next to the question far more reliably
    /// than one in the system prompt. Voice turns share the typed chat's
    /// system prompt so that, within a conversation, a spoken question
    /// reuses the engine's cached prefix of the typed turns before it (a
    /// different system prompt would change the very first tokens and
    /// nothing could be reused). The instruction is part of how a voice
    /// question is rendered every time, not only when it is asked, so the
    /// prefix of a later turn still matches.
    static let spokenReplyInstruction =
        "(Spoken aloud: answer in one or two short sentences of plain speech, with no formatting.)"

    /// Instructions for naming a conversation in the sidebar.
    static let titleSystemPrompt = """
        You name conversations. Reply with a short title of at most five \
        words and nothing else: no quotes, no punctuation at the end, no \
        explanation.
        """

    /// `base` with the user's custom instructions appended, when they have
    /// written any.
    static func systemPrompt(_ base: String, customInstructions: String) -> String {
        let custom = customInstructions.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !custom.isEmpty else { return base }
        return base + "\n\nThe user has given these instructions; follow them:\n" + custom
    }

    // MARK: - Generation presets

    /// A typed reply in Instant mode. Room for a few paragraphs or a code
    /// block; the engine stops early at the end of the answer.
    static let chatInstantOptions = ReplyOptions(maxTokens: 768, temperature: 0.7, topP: 0.95, think: false)

    /// A typed reply in Thinking mode: up to `thinkingBudget` tokens of
    /// reasoning, then the answer in what is left.
    static let chatThinkingOptions = ReplyOptions(maxTokens: 1024, temperature: 0.6, topP: 0.95, think: true)

    /// With thinking on, the most tokens the model may reason before the
    /// inferlet closes the reasoning and has it answer: about seven seconds
    /// at the phone's decode rate before the first word of the answer.
    static let thinkingBudget = 512

    /// A spoken reply. Short replies matter more here than in a text UI:
    /// every extra token is another second of someone waiting to be
    /// talked at, and reasoning would be spoken aloud as a stream of
    /// consciousness before the answer, so thinking stays off.
    static let voiceOptions = ReplyOptions(maxTokens: 160, temperature: 0.7, topP: 0.95, think: false)

    /// A conversation title: a handful of tokens, sampled conservatively.
    static let titleOptions = ReplyOptions(maxTokens: 16, temperature: 0.3, topP: 0.9, think: false)

    static func chatOptions(for mode: ReplyMode) -> ReplyOptions {
        switch mode {
        case .instant: return chatInstantOptions
        case .thinking: return chatThinkingOptions
        }
    }

    // MARK: - Token accounting

    /// Characters per token for ASCII text on Qwen's tokenizer, estimated:
    /// close for English prose and on the generous side for code.
    static let asciiCharactersPerToken = 3.2

    /// Estimated tokens in `text`, without running the tokenizer. ASCII
    /// counts at `asciiCharactersPerToken`; every other character counts
    /// as a whole token, which is about right for Chinese, Japanese and
    /// Korean (one character, one token) and errs on the safe side for
    /// accented Latin.
    static func estimatedTokens(in text: String) -> Int {
        Int(cost(of: text.unicodeScalars).rounded(.up))
    }

    /// The longest start of `text` estimated at no more than `tokens`.
    /// Never splits a character.
    static func prefix(of text: String, tokens: Int) -> String {
        var spent = 0.0
        var end = text.startIndex
        while end < text.endIndex {
            spent += cost(of: text[end].unicodeScalars)
            if spent > Double(tokens) { break }
            end = text.index(after: end)
        }
        return String(text[..<end])
    }

    /// The longest end of `text` estimated at no more than `tokens`.
    static func suffix(of text: String, tokens: Int) -> String {
        var spent = 0.0
        var start = text.endIndex
        while start > text.startIndex {
            let previous = text.index(before: start)
            spent += cost(of: text[previous].unicodeScalars)
            if spent > Double(tokens) { break }
            start = previous
        }
        return String(text[start...])
    }

    private static func cost<Scalars: Sequence>(of scalars: Scalars) -> Double
        where Scalars.Element == Unicode.Scalar {
        scalars.reduce(0) { $0 + ($1.isASCII ? 1 / asciiCharactersPerToken : 1) }
    }

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
    static let pieVersion = "Pie 0.5"

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
        max_model_len = \(contextTokens)
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
}
