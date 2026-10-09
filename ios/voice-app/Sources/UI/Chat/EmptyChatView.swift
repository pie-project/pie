import SwiftUI

/// A new chat before anything is sent: ChatGPT's centered question, in the
/// site's serif, with a reminder of where the model runs.
struct EmptyChatView: View {
    @EnvironmentObject private var chat: ChatController

    @ScaledMetric(relativeTo: .title) private var headlineSize: CGFloat = 28

    var body: some View {
        VStack(spacing: 10) {
            Text(chat.conversation.isTemporary ? "Temporary Chat" : "What can I help with?")
                .font(Theme.serif(headlineSize))
                .foregroundStyle(Theme.ink)
                .multilineTextAlignment(.center)
                .accessibilityAddTraits(.isHeader)
            if chat.conversation.isTemporary {
                Text("This chat won't appear in your history.")
            } else {
                Label("Runs entirely on this iPhone", systemImage: "lock.fill")
            }
        }
        .font(.footnote)
        .foregroundStyle(Theme.secondaryInk)
        .multilineTextAlignment(.center)
        .padding(.horizontal, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onTapGesture {
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
    }
}
