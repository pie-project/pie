import UIKit

/// The chat's haptics around a reply, as ChatGPT plays them: a tap when a
/// message goes out, and the soft ticks while the reply streams (the
/// first one when it starts; the controller ticks for every revealed
/// word). Nothing when it finishes. Each is silent when the user turned
/// haptics off. Built on `Haptics`, whose generators are prepared.
@MainActor
enum ChatHaptics {
    static func messageSent(enabled: Bool) {
        Haptics.tap(enabled: enabled)
    }

    /// The first tick of the streaming train; `Haptics.streamTick`
    /// throttles it against the ticks that follow.
    static func replyStarted(enabled: Bool) {
        Haptics.streamTick(enabled: enabled)
    }

    /// Silent on purpose: ChatGPT plays no success haptic when a reply
    /// completes. Kept so callers need not change.
    static func replyFinished(enabled: Bool) {}
}
