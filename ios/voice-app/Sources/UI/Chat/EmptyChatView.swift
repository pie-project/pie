import SwiftUI

/// A new chat before anything is sent: ChatGPT's centered question, in the
/// site's serif, with a reminder of where the model runs.
///
/// It fades in rising a few points when it appears, as ChatGPT's greeting
/// does, and turning Temporary Chat on or off crossfades the words in
/// place.
struct EmptyChatView: View {
    @EnvironmentObject private var chat: ChatController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @ScaledMetric(relativeTo: .title) private var headlineSize: CGFloat = 28
    @State private var hasAppeared = false

    var body: some View {
        let isTemporary = chat.conversation.isTemporary
        VStack(spacing: 10) {
            Text(isTemporary ? "Temporary Chat" : "What can I help with?")
                .font(Theme.serif(headlineSize))
                .foregroundStyle(Theme.ink)
                .multilineTextAlignment(.center)
                .contentTransition(.opacity)
                .accessibilityAddTraits(.isHeader)
            ZStack {
                if isTemporary {
                    Text("This chat won't appear in your history.")
                        .transition(.opacity)
                } else {
                    Label("Runs entirely on this iPhone", systemImage: "lock.fill")
                        .transition(.opacity)
                }
            }
        }
        // Local to the greeting: only its words change.
        .animation(Motion.crossfade, value: isTemporary)
        .font(.footnote)
        .foregroundStyle(Theme.secondaryInk)
        .multilineTextAlignment(.center)
        .padding(.horizontal, 32)
        .opacity(hasAppeared ? 1 : 0)
        .offset(y: hasAppeared || reduceMotion ? 0 : 6)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onTapGesture {
            KeyboardDismissal.dismiss()
        }
        .onAppear {
            withAnimation(Motion.fadeIn) { hasAppeared = true }
        }
    }
}
