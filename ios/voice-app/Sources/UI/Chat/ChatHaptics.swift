import UIKit

/// The chat's three haptics, as ChatGPT plays them: a tap when a message
/// goes out, a softer one when the reply starts to stream, and a success
/// tick when it is done. Each is silent when the user turned haptics off.
@MainActor
enum ChatHaptics {
    static func messageSent(enabled: Bool) {
        guard enabled else { return }
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    static func replyStarted(enabled: Bool) {
        guard enabled else { return }
        UIImpactFeedbackGenerator(style: .soft).impactOccurred()
    }

    static func replyFinished(enabled: Bool) {
        guard enabled else { return }
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }
}
