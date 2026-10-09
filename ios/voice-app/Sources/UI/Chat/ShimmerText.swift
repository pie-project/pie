import SwiftUI

/// Text with a highlight sweeping across it, ChatGPT's "Thinking" label.
///
/// The highlight is a repeating pattern, a soft band then a gap, one
/// `period` long, drawn twice side by side and slid one period to the
/// right per `Motion.shimmerPeriod`. Slid by a whole period the pattern
/// looks exactly as it started, so the loop has no visible restart: as
/// one band leaves on the right the next is already coming in on the
/// left. With Reduce Motion the label is static.
struct ShimmerText: View {
    let text: String

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Text(text)
            .foregroundStyle(Theme.tertiaryInk)
            .overlay {
                if !reduceMotion {
                    GeometryReader { proxy in
                        let band = proxy.size.width * 0.5
                        let period = proxy.size.width + band
                        HStack(spacing: 0) {
                            Self.pattern(band: band, period: period)
                            Self.pattern(band: band, period: period)
                        }
                        .frame(width: period * 2, alignment: .leading)
                        .keyframeAnimator(initialValue: CGFloat(0), repeating: true) { content, progress in
                            content.offset(x: -period + progress * period)
                        } keyframes: { _ in
                            LinearKeyframe(CGFloat(1), duration: Motion.shimmerPeriod)
                        }
                    }
                    .mask(Text(text))
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
                }
            }
    }

    /// One period of the sweep: the bright band, then empty space.
    private static func pattern(band: CGFloat, period: CGFloat) -> some View {
        HStack(spacing: 0) {
            LinearGradient(
                colors: [Theme.ink.opacity(0), Theme.ink, Theme.ink.opacity(0)],
                startPoint: .leading,
                endPoint: .trailing
            )
            .frame(width: band)
            Color.clear.frame(width: period - band)
        }
    }
}
