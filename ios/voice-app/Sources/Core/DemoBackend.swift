import Foundation

/// `-PieDemoBackend 1`: a stand-in model that streams canned replies at
/// the phone's real pace, so the interface and its animations can be
/// exercised and recorded where the engine cannot run (the Simulator) or
/// without waiting on it. Never used unless that argument is given.
///
/// Replies are cut into word-sized chunks and paced like the engine on an
/// iPhone 16 Pro: about 0.35 s to the first token, then roughly 40 words a
/// second, with the jitter real decoding has. Thinking mode streams a few
/// lines of reasoning first. `cancel(_:)` stops a reply between chunks.
final class DemoBackend: ConversationBackend {

    static var isEnabled: Bool {
        UserDefaults.standard.string(forKey: "PieDemoBackend") == "1"
    }

    var engineDescription: String { "Demo replies (no model)" }

    private let lock = NSLock()
    private var cancelled: Set<UInt64> = []

    func warmUp() async -> String? {
        try? await Task.sleep(nanoseconds: 800_000_000)
        return nil
    }

    func reply(
        _ ticket: ReplyTicket,
        session: String,
        messages: [PromptMessage],
        options: ReplyOptions,
        onEvent: @escaping (ReplyEvent) -> Void
    ) async throws -> ReplyResult {
        let started = Date()
        var stats = TurnStats()
        let question = messages.last { $0.role == .user }?.content ?? ""
        let short = options.maxTokens <= 200 || question.lowercased().contains("short")
        let answer = short ? Self.shortAnswer : Self.longAnswer

        try? await Task.sleep(nanoseconds: 350_000_000)
        var reasoning = ""
        if options.think {
            for chunk in Self.chunks(of: Self.reasoning) {
                if isCancelled(ticket) { break }
                if stats.timeToFirstToken == nil { stats.timeToFirstToken = Date().timeIntervalSince(started) }
                reasoning += chunk
                onEvent(.reasoning(chunk))
                await Self.pace()
            }
        }
        var text = ""
        for chunk in Self.chunks(of: answer) {
            if isCancelled(ticket) { break }
            if stats.timeToFirstToken == nil { stats.timeToFirstToken = Date().timeIntervalSince(started) }
            text += chunk
            stats.generated += 1
            onEvent(.text(chunk))
            await Self.pace()
        }
        let wasCancelled = isCancelled(ticket)
        forget(ticket)
        stats.promptTokens = messages.reduce(0) { $0 + $1.content.count / 4 }
        stats.newPrefill = stats.promptTokens
        stats.elapsed = Date().timeIntervalSince(started)
        return ReplyResult(text: text, reasoning: reasoning, stats: stats, cancelled: wasCancelled)
    }

    func cancel(_ ticket: ReplyTicket) {
        lock.lock()
        cancelled.insert(ticket.id)
        lock.unlock()
    }

    private func forget(_ ticket: ReplyTicket) {
        lock.lock()
        cancelled.remove(ticket.id)
        lock.unlock()
    }

    private func isCancelled(_ ticket: ReplyTicket) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled.contains(ticket.id)
    }

    /// One to three words per chunk, keeping the spaces and newlines, the
    /// way a tokenizer hands out text.
    private static func chunks(of text: String) -> [String] {
        var pieces: [String] = []
        var current = ""
        var words = 0
        var target = 1
        var index = 0
        for character in text {
            current.append(character)
            if character == " " || character == "\n" {
                words += 1
                if words >= target {
                    pieces.append(current)
                    current = ""
                    words = 0
                    index += 1
                    target = [1, 2, 1, 3, 1, 2][index % 6]
                }
            }
        }
        if !current.isEmpty { pieces.append(current) }
        return pieces
    }

    private static func pace() async {
        let jitter = UInt64.random(in: 0...30_000_000)
        try? await Task.sleep(nanoseconds: 25_000_000 + jitter)
    }

    static let reasoning = """
    The user wants a clear explanation. I should start with the short answer, \
    then give the steps, then a small example they can run. Keep it friendly \
    and skip anything they did not ask about.
    """

    static let shortAnswer = "Running the model on the phone keeps your words on the phone, and it still works with no signal."

    static let longAnswer = """
    Here is the short version: **an on-device model answers without sending anything to a server.**

    ## Why it matters

    - **Privacy.** Your questions and audio never leave the phone.
    - **Speed.** There is no network round trip, so the first word shows up fast.
    - **Offline.** It keeps working on a plane or in a basement.

    ## How it works here

    1. The app loads a small language model into the GPU's memory.
    2. Your message is turned into tokens and fed to the model.
    3. The model predicts one token at a time, and each one is streamed to the screen.

    A tiny example of streaming in Swift:

    ```swift
    for await token in model.generate(prompt) {
        reply += token
    }
    ```

    | Setting | Phone | Cloud |
    | --- | --- | --- |
    | Works offline | Yes | No |
    | First word | ~0.3 s | ~0.8 s |

    Want me to go deeper on any of these?
    """
}
