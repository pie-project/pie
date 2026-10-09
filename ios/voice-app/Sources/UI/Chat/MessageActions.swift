import SwiftUI

/// What a message's controls do, handed down by the message list.
///
/// The rows deliberately do not observe the controller: one that did would
/// be redrawn on every streamed token, and so would every other row in the
/// conversation. They take plain values plus these closures and compare
/// equal when their values have not changed.
struct MessageActions {
    let setFeedback: (Feedback?, UUID) -> Void
    let toggleReadAloud: (UUID) -> Void
    let regenerate: (UUID, ReplyMode?) -> Void
}

extension EnvironmentValues {
    /// Whether a reply can be regenerated now (the engine is ready and
    /// nothing is being generated). An environment value rather than a
    /// row input: it flips at every send and finish, and only the
    /// Regenerate controls that read it are redrawn, not every reply.
    var canRegenerate: Bool {
        get { self[CanRegenerateKey.self] }
        set { self[CanRegenerateKey.self] = newValue }
    }
}

private struct CanRegenerateKey: EnvironmentKey {
    static let defaultValue = true
}
