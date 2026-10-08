import Foundation

/// `ConversationBackend` backed by the embedded Pie engine.
///
/// The shim boots the engine once per process and keeps it warm, and
/// `pie_ios_run_stream` blocks for the length of a turn, so every call is
/// serialised onto one background queue. Nothing above this class needs
/// to know either fact.
final class PieEngine: ConversationBackend {

    private let queue = DispatchQueue(label: "org.pie-project.voice.engine", qos: .userInitiated)
    private var didWarmUp = false

    var engineDescription: String {
        "\(PieRuntimeConfig.modelDescription) · \(PieRuntimeConfig.driverDescription) · inferlets on \(PieRuntimeConfig.runtimeDescription)"
    }

    // MARK: - ConversationBackend

    func warmUp() async -> String? {
        guard !didWarmUp else { return nil }
        didWarmUp = true
        print("[warmup] engine boot starting")
        // Boots the engine and loads the model weights by running a
        // single-token turn. The snapshot it leaves behind is discarded:
        // the first real turn starts fresh and overwrites it.
        do {
            _ = try await run(
                text: "hello",
                startingFresh: true,
                maxTokens: 1,
                onDelta: { _ in }
            )
            print("[warmup] engine boot complete")
            return nil
        } catch {
            // Swallowing this is how a boot failure turns into a silent
            // forever-spinner. Surface it.
            print("[warmup] engine boot FAILED: \(error)")
            didWarmUp = false
            return String(describing: error)
        }
    }

    func reply(
        to utterance: String,
        startingFresh: Bool,
        onDelta: @escaping (String) -> Void
    ) async throws -> (text: String, stats: TurnStats) {
        try await run(
            text: utterance,
            startingFresh: startingFresh,
            maxTokens: PieRuntimeConfig.maxTokensPerTurn,
            onDelta: onDelta
        )
    }

    // MARK: - Invocation

    private func run(
        text: String,
        startingFresh: Bool,
        maxTokens: Int,
        onDelta: @escaping (String) -> Void
    ) async throws -> (text: String, stats: TurnStats) {
        let inferlet = PieRuntimeConfig.voiceChat
        let input = Self.inputJSON(
            text: text,
            startingFresh: startingFresh,
            maxTokens: maxTokens
        )

        // The closure below runs on `queue` and deliberately captures no
        // mutable state: the engine's own warmth is what makes repeat
        // calls cheap, so there is nothing worth caching on this side.
        let queue = self.queue

        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let config = try PieRuntimeConfig.writeEngineConfig()
                    let started = Date()
                    let result = PieBridge.runStreaming(
                        configPath: config,
                        wasmPath: inferlet.wasmPath,
                        manifestPath: inferlet.manifestPath,
                        inferletId: inferlet.id,
                        inputJSON: input,
                        onDelta: onDelta
                    )
                    let elapsed = -started.timeIntervalSinceNow

                    if result.hasPrefix("PIE ERROR") {
                        continuation.resume(throwing: ConversationError.backend(result))
                        return
                    }
                    continuation.resume(returning: PieEngine.parse(result, elapsed: elapsed))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Wire format

    private static func inputJSON(
        text: String,
        startingFresh: Bool,
        maxTokens: Int
    ) -> String {
        let payload: [String: Any] = [
            "text": text,
            "session": PieRuntimeConfig.sessionName,
            "reset": startingFresh,
            "max_tokens": maxTokens,
            "temperature": PieRuntimeConfig.temperature,
            "top_p": PieRuntimeConfig.topP,
        ]
        guard
            let data = try? JSONSerialization.data(withJSONObject: payload),
            let json = String(data: data, encoding: .utf8)
        else {
            return "{\"text\":\"\"}"
        }
        return json
    }

    /// The inferlet returns its reply and KV accounting as JSON so the
    /// numbers never land on the spoken stdout channel.
    private static func parse(_ result: String, elapsed: TimeInterval) -> (text: String, stats: TurnStats) {
        var stats = TurnStats()
        stats.elapsed = elapsed

        guard
            let data = result.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            // An inferlet that returned something unexpected still said
            // something usable — treat the raw string as the reply.
            return (result.trimmingCharacters(in: .whitespacesAndNewlines), stats)
        }

        stats.reused = object["reused"] as? Int ?? 0
        stats.newPrefill = object["new_prefill"] as? Int ?? 0
        stats.generated = object["generated"] as? Int ?? 0
        stats.resumed = object["resumed"] as? Bool ?? false
        stats.note = object["note"] as? String ?? ""
        let text = (object["text"] as? String ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (text, stats)
    }
}
