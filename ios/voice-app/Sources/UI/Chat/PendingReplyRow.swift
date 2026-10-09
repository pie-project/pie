import SwiftUI

/// Stands in for a reply the controller has not added to the transcript
/// yet: the pulsing dot, or "Thinking" once reasoning has started.
struct PendingReplyRow: View {
    let phase: ChatController.ReplyPhase

    var body: some View {
        Group {
            if case .thinking = phase {
                ShimmerText(text: "Thinking")
                    .font(.subheadline.weight(.medium))
                    .frame(minHeight: 32)
            } else {
                PulsingDot()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
