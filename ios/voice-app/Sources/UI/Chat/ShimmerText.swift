import SwiftUI

/// Text with a highlight sweeping across it, ChatGPT's "Thinking" label.
struct ShimmerText: View {
    let text: String

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Text(text)
            .foregroundStyle(Theme.tertiaryInk)
            .overlay {
                if !reduceMotion {
                    GeometryReader { proxy in
                        let width = proxy.size.width
                        LinearGradient(
                            colors: [Theme.ink.opacity(0), Theme.ink, Theme.ink.opacity(0)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                        .frame(width: width * 0.6)
                        .keyframeAnimator(initialValue: CGFloat(0), repeating: true) { content, progress in
                            content.offset(x: -width * 0.6 + progress * width * 1.6)
                        } keyframes: { _ in
                            LinearKeyframe(CGFloat(1), duration: 1.4)
                        }
                    }
                    .mask(Text(text))
                }
            }
    }
}
