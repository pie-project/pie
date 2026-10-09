import SwiftUI

/// The reasoning above a thinking-mode reply: a shimmering "Thinking"
/// while it streams, then "Thought for Ns". Either one expands to show the
/// reasoning itself, set off by a rule in the accent color.
///
/// "Thinking" crossfades into "Thought for Ns" in place (the controller
/// makes that change animated), and the chevron glides to the new label's
/// end rather than jumping. Opening rotates the chevron and reveals the
/// reasoning from the top down as the reply below moves to make room; its
/// text fades in just behind the reveal, so fading text never overlaps
/// moving text. Closing runs the same reveal backwards.
struct ReasoningDisclosure: View {
    let reasoning: String
    let isThinking: Bool
    let thoughtSeconds: Double?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withMotion(Motion.content) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 4) {
                    ZStack(alignment: .leading) {
                        if isThinking {
                            ShimmerText(text: "Thinking")
                                .transition(.opacity)
                        } else {
                            Text(title)
                                .foregroundStyle(Theme.secondaryInk)
                                .transition(.opacity)
                        }
                    }
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.tertiaryInk)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .animation(reduceMotion ? nil : .snappy(duration: 0.25), value: isExpanded)
                }
                .font(.subheadline.weight(.medium))
                .frame(minHeight: 32)
                .contentShape(Rectangle())
            }
            .buttonStyle(PressDimButtonStyle())
            .accessibilityLabel(isThinking ? "Thinking" : title)
            .accessibilityHint(isExpanded ? "Hides the reasoning" : "Shows the reasoning")

            if isExpanded && !reasoning.isEmpty {
                Text(reasoning)
                    .font(.subheadline)
                    .foregroundStyle(Theme.secondaryInk)
                    .lineSpacing(3)
                    .padding(.leading, 12)
                    .background(alignment: .leading) {
                        Rectangle().fill(Theme.accent).frame(width: 2)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .transition(Self.reveal)
            }
        }
    }

    private var title: String {
        guard let seconds = thoughtSeconds, seconds >= 1 else { return "Thought for a moment" }
        return "Thought for \(Int(seconds.rounded()))s"
    }

    /// Unmasked from the top down in step with the layout making room,
    /// with the text fading in a moment behind; the reverse on the way out,
    /// where the fade is quicker than the collapse.
    private static let reveal: AnyTransition = .asymmetric(
        insertion: .modifier(active: TopReveal(fraction: 0), identity: TopReveal(fraction: 1))
            .combined(with: .opacity.animation(Motion.fadeIn.delay(0.05))),
        removal: .modifier(active: TopReveal(fraction: 0), identity: TopReveal(fraction: 1))
            .combined(with: .opacity.animation(Motion.fadeOut))
    )
}

/// Shows only the top `fraction` of the content; animatable, so a
/// transition can grow or shrink the visible part smoothly.
private struct TopReveal: ViewModifier, Animatable {
    var fraction: CGFloat

    var animatableData: CGFloat {
        get { fraction }
        set { fraction = newValue }
    }

    func body(content: Content) -> some View {
        content.mask(alignment: .top) {
            Rectangle().scaleEffect(x: 1, y: fraction, anchor: .top)
        }
    }
}
