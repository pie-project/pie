import Foundation

/// The language model, seen from the conversation layer.
///
/// Deliberately says nothing about Pie, inferlets, wasm, or model
/// artifacts: the controllers and the views are written against this
/// protocol only, so the serving stack can be replaced or upgraded
/// without touching them.
///
/// The backend keeps no conversation of its own. Every call carries the
/// whole transcript; whether earlier turns are served from cached state or
/// prefilled again is the backend's business and shows up in the stats.
protocol ConversationBackend: AnyObject {
    /// One-line description of what is actually serving, for the UI.
    var engineDescription: String { get }

    /// Boot cost paid ahead of the first message. Safe to call twice.
    /// Returns nil on success, or a message saying why the boot failed.
    @discardableResult
    func warmUp() async -> String?

    /// Generates one reply.
    ///
    /// Replies run one at a time, in call order. `onEvent` may be called on
    /// any thread. A reply cancelled with `cancel(_:)` returns normally with
    /// `cancelled == true` and whatever text it had produced; it does not
    /// throw.
    ///
    /// - Parameters:
    ///   - ticket: names this reply for `cancel(_:)`.
    ///   - session: the prefix-cache namespace of the conversation, so
    ///     each thread reuses its own earlier turns.
    ///   - messages: the transcript, oldest first, ending with the new user
    ///     message; an optional leading system message.
    func reply(
        _ ticket: ReplyTicket,
        session: String,
        messages: [PromptMessage],
        options: ReplyOptions,
        onEvent: @escaping (ReplyEvent) -> Void
    ) async throws -> ReplyResult

    /// Stops the reply `ticket` names: immediately if it is generating,
    /// before it starts if it is still queued, and not at all if it has
    /// already returned. Returns at once; never blocks on the engine.
    func cancel(_ ticket: ReplyTicket)
}
