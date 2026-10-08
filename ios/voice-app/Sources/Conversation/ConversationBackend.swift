import Foundation

/// What one generated turn cost, as reported by the backend.
///
/// `reused` is the point of the whole exercise: tokens of conversation
/// that did not have to be prefilled again this turn.
struct TurnStats {
    var reused: Int = 0
    var newPrefill: Int = 0
    var generated: Int = 0
    var resumed: Bool = false
    var elapsed: TimeInterval = 0
    /// Set when the backend had to fall back to a slower path for this
    /// turn. Surfaced rather than swallowed: a demo that quietly degrades
    /// is worse than one that says it degraded.
    var note: String = ""

    var tokensPerSecond: Double {
        elapsed > 0 ? Double(generated) / elapsed : 0
    }
}

enum ConversationError: LocalizedError {
    case backend(String)

    var errorDescription: String? {
        switch self {
        case .backend(let message): return message
        }
    }
}

/// The language model, seen from the conversation layer.
///
/// Deliberately says nothing about Pie, inferlets, wasm, or GGUF: the
/// controller and the views are written against this protocol only, so
/// the serving stack can be replaced or upgraded without touching them.
protocol ConversationBackend: AnyObject {
    /// One-line description of what is actually serving, for the UI.
    var engineDescription: String { get }

    /// Boot cost paid ahead of the first utterance. Safe to call twice.
    /// Returns nil on success, or a message describing why boot failed.
    @discardableResult
    func warmUp() async -> String?

    /// Generates a reply to one utterance.
    ///
    /// - Parameters:
    ///   - utterance: what the user just said.
    ///   - startingFresh: discard any prior conversation state first.
    ///   - onDelta: speakable text, as it is produced. May be called on
    ///     any thread.
    /// - Returns: the complete reply and what the turn cost.
    func reply(
        to utterance: String,
        startingFresh: Bool,
        onDelta: @escaping (String) -> Void
    ) async throws -> (text: String, stats: TurnStats)
}
