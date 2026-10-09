import SwiftUI

/// ChatGPT's "reply on its way" dot, pulsing until the first token.
struct PulsingDot: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Group {
            if reduceMotion {
                dot
            } else {
                dot.phaseAnimator([false, true]) { content, expanded in
                    content
                        .scaleEffect(expanded ? 1 : 0.7)
                        .opacity(expanded ? 1 : 0.55)
                } animation: { _ in
                    .easeInOut(duration: 0.65)
                }
            }
        }
        .frame(height: 24)
        .accessibilityElement()
        .accessibilityLabel("Pie is replying")
    }

    private var dot: some View {
        Circle()
            .fill(Theme.accent)
            .frame(width: 14, height: 14)
    }
}
