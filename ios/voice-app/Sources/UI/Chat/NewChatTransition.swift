import SwiftUI

/// Starting a new chat from a conversation, as ChatGPT does it: the
/// conversation fades out and the greeting fades in; nothing slides.
///
/// It is two short fades rather than one crossfade because the transcript
/// draws the open conversation as it is now. Replaced by an empty one
/// while it was still fading, the outgoing transcript lost its rows and
/// jumped to its top (measured in the Simulator). So the transcript fades
/// out first, and the conversation is swapped once it is out of sight.
@MainActor
final class NewChatTransition: ObservableObject {
    /// The transcript has faded out for the swap. `ChatScreen` reads it.
    @Published private(set) var isTranscriptFadedOut = false

    /// `Motion.fadeOut`'s length: the swap waits for the fade to finish.
    private static let fadeOutDuration: TimeInterval = 0.14

    /// Leaves the open conversation for a new one. `beforeSwap` runs at
    /// the moment of the swap (deleting the conversation being left, say).
    func start(_ chat: ChatController, temporary: Bool = false, beforeSwap: (() -> Void)? = nil) {
        // One already under way.
        guard !isTranscriptFadedOut else { return }
        // An empty chat has no transcript to lose: the greeting simply
        // crossfades (to "Temporary Chat", say).
        guard !chat.conversation.isEmpty else {
            beforeSwap?()
            withMotion(Motion.crossfade) {
                chat.newChat(temporary: temporary)
            }
            return
        }
        withAnimation(Motion.fadeOut) {
            isTranscriptFadedOut = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.fadeOutDuration) {
            beforeSwap?()
            // The faded-out transcript is simply removed; the greeting and
            // the chips fade in, and the bar's items crossfade.
            withAnimation(Motion.fadeIn) {
                chat.newChat(temporary: temporary)
                self.isTranscriptFadedOut = false
            }
        }
    }
}
