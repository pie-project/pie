import Foundation

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
