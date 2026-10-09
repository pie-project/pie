import SwiftUI

/// Starting a new chat from a conversation, as ChatGPT does it: one short
/// crossfade (`Motion.crossfade`, 0.22 s) from the transcript to the
/// greeting; nothing slides.
///
/// The conversation is not swapped when the fade starts. The transcript
/// draws the open conversation as it is now, and replaced by an empty one
/// while it was still fading it lost its rows and jumped to its top
/// (measured in the Simulator). So the greeting fades in over the
/// transcript while the transcript fades out, both drawn from the old
/// conversation; once the transcript is invisible, the conversation is
/// swapped behind the greeting, which is already in place. (The first
/// version faded the transcript out first and the greeting in after it,
/// and the screen went blank for a moment between the two.)
///
/// For those 0.22 s (`isCrossfading`) the chat and the composer ignore
/// touches, so nothing lands in the conversation being left. A second
/// request in that time joins this one rather than being dropped: its
/// action (deleting the chat being left) runs at the swap, and the last
/// request decides whether the new chat is temporary. If the open
/// conversation changed meanwhile (a chat picked in the sidebar), that
/// chat fades back in instead of being replaced.
@MainActor
final class NewChatTransition: ObservableObject {
    /// The transcript is fading out with the greeting fading in over it.
    /// `ChatScreen` reads it.
    @Published private(set) var isCrossfading = false
    /// Whether the chat on its way in is temporary, so that the greeting
    /// fading in already says "Temporary Chat" (`EmptyChatView`).
    @Published private(set) var incomingIsTemporary = false

    /// `Motion.crossfade`'s length: the swap waits until the transcript is
    /// out of sight.
    private static let crossfadeDuration: TimeInterval = 0.22

    /// The conversation being left.
    private var leaving: UUID?
    /// What to do at the swap (delete the conversation being left).
    private var beforeSwap: [() -> Void] = []

    /// Leaves the open conversation for a new one. `beforeSwap` runs at
    /// the moment of the swap.
    func start(_ chat: ChatController, temporary: Bool = false, beforeSwap action: (() -> Void)? = nil) {
        if let action { beforeSwap.append(action) }
        if isCrossfading {
            // One is under way: this request joins it.
            if temporary != incomingIsTemporary {
                withMotion(Motion.crossfade) { incomingIsTemporary = temporary }
            }
            return
        }
        // An empty chat has no transcript to lose: the greeting simply
        // crossfades in place (to "Temporary Chat", say).
        guard !chat.conversation.isEmpty else {
            runBeforeSwap()
            withMotion(Motion.crossfade) {
                chat.newChat(temporary: temporary)
            }
            return
        }
        leaving = chat.conversation.id
        withMotion(Motion.crossfade) {
            incomingIsTemporary = temporary
            isCrossfading = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.crossfadeDuration) { [weak chat] in
            guard let chat else { return }
            self.swap(chat)
        }
    }

    /// The transcript is invisible now: the new chat takes its place
    /// behind the greeting. The faded-out transcript is simply removed,
    /// and the top bar's items crossfade to the empty chat's.
    private func swap(_ chat: ChatController) {
        runBeforeSwap()
        withMotion(Motion.crossfade) {
            if chat.conversation.id == leaving {
                chat.newChat(temporary: incomingIsTemporary)
            }
            isCrossfading = false
        }
        leaving = nil
    }

    private func runBeforeSwap() {
        let actions = beforeSwap
        beforeSwap = []
        actions.forEach { $0() }
    }
}
