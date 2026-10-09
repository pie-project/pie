import SwiftUI

/// A new chat before anything is sent: ChatGPT's centered question, in the
/// site's serif, with a reminder of where the model runs.
///
/// It has no entrance of its own: whoever puts it on screen fades it in
/// (`ChatScreen`; a new chat's crossfade). It used to fade itself in as
/// well, and the two fades stacked, so the greeting stayed almost
/// invisible for the first part of a new chat (recorded).
///
/// Turning Temporary Chat on or off swaps the words in two overlapping
/// steps rather than one crossfade: the old words fade out quickly and the
/// new ones fade in a beat later. Crossfaded in place, the two serif
/// headlines showed on top of each other at half strength for 150 ms,
/// their letters interleaved (recorded).
///
/// Both versions are always laid out, one of them invisible, and only
/// their opacity changes, each on its own timing (`.animation(_:value:)`).
/// Swapped in and out with transitions instead, the old words kept the
/// toggle's longer fade and still overlapped the new ones for about 80 ms
/// (recorded). Laid out together, the greeting also keeps one size, so the
/// lines under it do not move.
struct EmptyChatView: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var newChat: NewChatTransition

    @ScaledMetric(relativeTo: .title) private var headlineSize: CGFloat = 28

    var body: some View {
        // While a new chat crossfades in, the greeting already shows the
        // chat on its way, not the one being left.
        let isTemporary = newChat.isCrossfading ? newChat.incomingIsTemporary : chat.conversation.isTemporary
        VStack(spacing: 10) {
            ZStack {
                swapping(headline("What can I help with?"), shown: !isTemporary)
                swapping(headline("Temporary Chat"), shown: isTemporary)
            }
            ZStack {
                swapping(Label("Runs entirely on this iPhone", systemImage: "lock.fill"), shown: !isTemporary)
                swapping(Text("This chat won't appear in your history."), shown: isTemporary)
            }
        }
        .font(.footnote)
        .foregroundStyle(Theme.secondaryInk)
        .multilineTextAlignment(.center)
        .padding(.horizontal, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onTapGesture {
            KeyboardDismissal.dismiss()
        }
    }

    private func headline(_ text: String) -> some View {
        Text(text)
            .font(Theme.serif(headlineSize))
            .foregroundStyle(Theme.ink)
            .accessibilityAddTraits(.isHeader)
    }

    /// Words that leave in `Motion.fadeOut` (0.14 s) and arrive once the
    /// other words are gone, so the two headlines are never drawn on top of
    /// each other (half way, they were, for a few frames); hidden from
    /// VoiceOver while out.
    private func swapping<Content: View>(_ content: Content, shown: Bool) -> some View {
        content
            .opacity(shown ? 1 : 0)
            .animation(shown ? Motion.fadeIn.delay(0.12) : Motion.fadeOut, value: shown)
            .accessibilityHidden(!shown)
    }
}
