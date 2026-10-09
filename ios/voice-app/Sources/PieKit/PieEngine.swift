import Foundation

/// `ConversationBackend` backed by the embedded Pie engine.
///
/// The shim boots the engine once per process and keeps it warm, and a
/// run blocks its thread for the length of a reply, so every reply is
/// serialised onto one background queue. Cancelling goes around that
/// queue on purpose: the shim's cancel is thread-safe and terminates the
/// engine process in flight, which is what frees the queue for the reply
/// waiting behind it. Routing a cancel through the queue would park it
/// behind the very reply it is meant to stop.
final class PieEngine: ConversationBackend {

    private let queue = DispatchQueue(label: "org.pie-project.voice.engine", qos: .userInitiated)
    private let warmLock = NSLock()
    private var isWarm = false

    init() {}

    var engineDescription: String {
        "\(PieRuntimeConfig.modelDescription) · \(PieRuntimeConfig.driverDescription) · inferlets on \(PieRuntimeConfig.runtimeDescription)"
    }

    // MARK: - ConversationBackend

    func warmUp() async -> String? {
        if warmed { return nil }
        print("[warmup] engine boot starting")
        // Boots the engine and loads the model weights by running a
        // single-token turn on a throwaway session, so whatever state it
        // leaves in the engine is never resumed by a real conversation.
        do {
            _ = try await reply(
                ReplyTicket(),
                session: PieRuntimeConfig.warmUpSessionName,
                messages: [
                    PromptMessage(role: .system, content: PieRuntimeConfig.chatSystemPrompt),
                    PromptMessage(role: .user, content: "hello"),
                ],
                options: ReplyOptions(maxTokens: 1, temperature: 0.7, topP: 0.95, think: false),
                onEvent: { _ in }
            )
            warmed = true
            print("[warmup] engine boot complete")
            return nil
        } catch {
            // Swallowing this is how a boot failure turns into a silent
            // forever-spinner. Surface it. Both failures that can land
            // here (a missing artifact, a "PIE ERROR" from the shim) carry
            // their message in `errorDescription`; interpolating the error
            // itself would show the enum case and payload instead.
            let message = error.localizedDescription
            print("[warmup] engine boot FAILED: \(message)")
            return message
        }
    }

    func reply(
        _ ticket: ReplyTicket,
        session: String,
        messages: [PromptMessage],
        options: ReplyOptions,
        onEvent: @escaping (ReplyEvent) -> Void
    ) async throws -> ReplyResult {
        let input = Self.inputJSON(
            session: session,
            messages: PromptBudget.fit(messages, maxTokens: options.maxTokens),
            options: options
        )
        let turnID = ticket.id
        let queue = self.queue

        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(with: Result {
                    try Self.run(turnID: turnID, inputJSON: input, onEvent: onEvent)
                })
            }
        }
    }

    func cancel(_ ticket: ReplyTicket) {
        PieBridge.cancel(turnID: ticket.id)
    }

    // MARK: - Invocation

    private var warmed: Bool {
        get {
            warmLock.lock()
            defer { warmLock.unlock() }
            return isWarm
        }
        set {
            warmLock.lock()
            isWarm = newValue
            warmLock.unlock()
        }
    }

    /// One reply, start to finish, on the engine queue. The clock starts
    /// here rather than when `reply` was called, so a reply that waited
    /// behind another is not charged for the wait.
    private static func run(
        turnID: UInt64,
        inputJSON: String,
        onEvent: @escaping (ReplyEvent) -> Void
    ) throws -> ReplyResult {
        let config = try PieRuntimeConfig.writeEngineConfig()
        let inferlet = PieRuntimeConfig.voiceChat
        let started = Date()

        // Touched only from inside `PieBridge.run`, on this thread.
        var streamedText = ""
        var streamedReasoning = ""
        var firstEventAt: Date?

        let outcome = PieBridge.run(
            configPath: config,
            wasmPath: inferlet.wasmPath,
            version: inferlet.version,
            inputJSON: inputJSON,
            turnID: turnID
        ) { channel, chunk in
            if firstEventAt == nil { firstEventAt = Date() }
            switch channel {
            case .reply:
                streamedText += chunk
                onEvent(.text(chunk))
            case .reasoning:
                streamedReasoning += chunk
                onEvent(.reasoning(chunk))
            }
        }

        var stats = TurnStats()
        stats.elapsed = -started.timeIntervalSinceNow
        stats.timeToFirstToken = firstEventAt.map { $0.timeIntervalSince(started) }

        switch outcome {
        case .failed(let message):
            throw ConversationError.backend(message)
        case .cancelled:
            // The engine process was terminated, so there is no return
            // value to read; what streamed is all there is.
            return ReplyResult(
                text: streamedText.trimmingCharacters(in: .whitespacesAndNewlines),
                reasoning: streamedReasoning.trimmingCharacters(in: .whitespacesAndNewlines),
                stats: stats,
                cancelled: true
            )
        case .completed(let returned):
            return parse(
                returned,
                streamedText: streamedText,
                streamedReasoning: streamedReasoning,
                stats: stats
            )
        }
    }

    // MARK: - Wire format

    /// The inferlet takes the whole transcript every turn. What it can
    /// serve from the session's cached state is its own accounting,
    /// returned in stats.
    private static func inputJSON(
        session: String,
        messages: [PromptMessage],
        options: ReplyOptions
    ) -> String {
        var payload: [String: Any] = [
            "messages": messages.map { ["role": $0.role.rawValue, "content": $0.content] },
            "session": session,
            "max_tokens": options.maxTokens,
            "temperature": options.temperature,
            "top_p": options.topP,
            "think": options.think,
        ]
        if options.think {
            payload["thinking_budget"] = PieRuntimeConfig.thinkingBudget
        }
        guard
            let data = try? JSONSerialization.data(withJSONObject: payload),
            let json = String(data: data, encoding: .utf8)
        else {
            return "{\"messages\":[]}"
        }
        return json
    }

    /// The inferlet returns its reply and token accounting as JSON so the
    /// numbers never land on the streamed text channel.
    private static func parse(
        _ returned: String,
        streamedText: String,
        streamedReasoning: String,
        stats: TurnStats
    ) -> ReplyResult {
        var stats = stats
        guard
            let data = returned.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            // An inferlet that returned something unexpected still said
            // something usable: prefer what streamed, else the raw value.
            let text = streamedText.isEmpty ? returned : streamedText
            return ReplyResult(
                text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                reasoning: streamedReasoning.trimmingCharacters(in: .whitespacesAndNewlines),
                stats: stats,
                cancelled: false
            )
        }

        stats.promptTokens = object["prompt_tokens"] as? Int ?? 0
        stats.reused = object["reused"] as? Int ?? 0
        stats.newPrefill = object["new_prefill"] as? Int ?? 0
        stats.generated = object["generated"] as? Int ?? 0
        stats.resumed = object["resumed"] as? Bool ?? false
        stats.note = object["note"] as? String ?? ""

        let text = object["text"] as? String ?? streamedText
        let reasoning = (object["reasoning"] as? String).flatMap { $0.isEmpty ? nil : $0 }
            ?? streamedReasoning
        return ReplyResult(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            reasoning: reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            stats: stats,
            cancelled: false
        )
    }
}

/// Keeps a prompt inside the engine's per-sequence context.
///
/// The engine holds `PieRuntimeConfig.contextTokens` for prompt and reply
/// together, and a prompt past that fails the turn outright. A long chat
/// gets there, so the oldest exchanges are left out until the system
/// prompt, what remains of the history, the new message, the reply's
/// token budget and a margin for the chat template all fit. The cost is
/// that the model forgets the start of a very long conversation, which
/// beats refusing to answer.
///
/// History is cut only at fixed points, the first exchange boundary past
/// each `trimStep` tokens of it, never at whichever exchange happens to
/// make this one turn fit. The engine serves a turn from the state the
/// turn before left only when the new prompt starts with the old one.
/// Leaving out one more exchange every turn would change the start of the
/// prompt every turn, and every turn of a long chat would prefill
/// thousands of tokens again. Cut at a fixed point, the prompt keeps its
/// start until the history has grown by about another step.
///
/// A new message too long to fit even with no history at all is shortened
/// in the middle, keeping its start and its end, where the question
/// usually is.
enum PromptBudget {
    /// Role markers and turn delimiters the template wraps around each
    /// message, plus the generation cue and the tokens that close a
    /// reasoning block.
    static let templateMargin = 256
    /// How far apart the points history is cut at are, in tokens.
    static let trimStep = PieRuntimeConfig.contextTokens / 4

    private static let elision = "\n\n[The middle of this message was left out to fit.]\n\n"

    static func estimate(_ message: PromptMessage) -> Int {
        PieRuntimeConfig.estimatedTokens(in: message.content)
    }

    /// `messages` cut to fit: the leading system message always stays,
    /// whole user/assistant exchanges are left out from the oldest, and
    /// the final (new) message stays, shortened only if it cannot fit
    /// otherwise.
    static func fit(_ messages: [PromptMessage], maxTokens: Int) -> [PromptMessage] {
        guard let newest = messages.last else { return messages }
        var history = Array(messages.dropLast())
        var system: [PromptMessage] = []
        if history.first?.role == .system {
            system.append(history.removeFirst())
        }

        let room = PieRuntimeConfig.contextTokens - maxTokens - templateMargin
            - system.map(estimate).reduce(0, +)
        let newestCost = estimate(newest)
        guard newestCost <= room else {
            let target = max(room, 0)
            print("[prompt] shortened the new message from about \(newestCost) to \(target) tokens "
                + "and left out all \(history.count) earlier messages to fit \(PieRuntimeConfig.contextTokens) tokens")
            let shortened = PromptMessage(role: newest.role, content: shortened(newest.content, toTokens: target))
            return system + [shortened]
        }

        let kept = trimmed(history, toTokens: room - newestCost)
        if kept.count < history.count {
            print("[prompt] left out the \(history.count - kept.count) oldest messages to fit \(PieRuntimeConfig.contextTokens) tokens")
        }
        return system + kept + [newest]
    }

    /// `history` from the first cut point that leaves at most `room`
    /// tokens of it, or all of it if it fits.
    private static func trimmed(_ history: [PromptMessage], toTokens room: Int) -> [PromptMessage] {
        let costs = history.map(estimate)
        let total = costs.reduce(0, +)
        guard total > room else { return history }

        var dropped = 0
        var nextCut = trimStep
        var index = 0
        while index < history.count {
            let isExchange = index + 1 < history.count
                && history[index].role == .user
                && history[index + 1].role == .assistant
            let count = isExchange ? 2 : 1
            dropped += costs[index ..< index + count].reduce(0, +)
            index += count
            // A cut point is the first boundary at or past each step; the
            // boundaries in between are never cut at, however well they
            // would fit this turn.
            guard dropped >= nextCut || index == history.count else { continue }
            while nextCut <= dropped { nextCut += trimStep }
            if total - dropped <= room { break }
        }
        return Array(history[index...])
    }

    private static func shortened(_ text: String, toTokens tokens: Int) -> String {
        let half = max(tokens - PieRuntimeConfig.estimatedTokens(in: elision), 0) / 2
        return PieRuntimeConfig.prefix(of: text, tokens: half)
            + elision
            + PieRuntimeConfig.suffix(of: text, tokens: half)
    }
}
