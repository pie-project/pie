import SwiftUI

/// The reasoning above a thinking-mode reply: a shimmering "Thinking"
/// while it streams, then "Thought for Ns". Either one expands to show the
/// reasoning itself, set off by a rule in the accent color.
struct ReasoningDisclosure: View {
    let reasoning: String
    let isThinking: Bool
    let thoughtSeconds: Double?

    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 4) {
                    if isThinking {
                        ShimmerText(text: "Thinking")
                    } else {
                        Text(title).foregroundStyle(Theme.secondaryInk)
                    }
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.tertiaryInk)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                .font(.subheadline.weight(.medium))
                .frame(minHeight: 32)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
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
                    .transition(.opacity)
            }
        }
    }

    private var title: String {
        guard let seconds = thoughtSeconds, seconds >= 1 else { return "Thought for a moment" }
        return "Thought for \(Int(seconds.rounded()))s"
    }
}
