import SwiftUI

/// ChatGPT's "reply on its way" dot: one ink-colored dot where the reply's
/// first word will be, breathing until the first words replace it.
///
/// It arrives a beat after the question's bubble (a short scale and
/// fade), breathes between 70% and 100% over `Motion.pulsePeriod`, and is
/// faded out by its row as the first words fade in over it. With Reduce
/// Motion it only fades in and stays still.
struct PulsingDot: View {
    /// The height of the reply's first line, which the dot is centered on.
    static let lineHeight: CGFloat = 22
    private static let diameter: CGFloat = 13

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hasArrived = false

    var body: some View {
        Circle()
            .fill(Theme.ink)
            .frame(width: Self.diameter, height: Self.diameter)
            .phaseAnimator(reduceMotion ? [1.0] : [1.0, 0.7]) { dot, scale in
                dot.scaleEffect(scale)
            } animation: { _ in
                .easeInOut(duration: Motion.pulsePeriod / 2)
            }
            .scaleEffect(hasArrived || reduceMotion ? 1 : 0.5)
            .opacity(hasArrived ? 1 : 0)
            .frame(width: Self.diameter, height: Self.lineHeight)
            .onAppear {
                withAnimation(.easeOut(duration: 0.2).delay(0.1)) { hasArrived = true }
            }
            .accessibilityElement()
            .accessibilityLabel("Pie is replying")
    }
}
