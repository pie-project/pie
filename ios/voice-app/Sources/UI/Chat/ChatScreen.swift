import SwiftUI

/// The chat: the top bar, the engine's status, the conversation (or the
/// empty state and starter chips), and the composer pinned above the
/// keyboard.
///
/// The navigation stack never pushes anything. It is there for its bar,
/// the only kind of top bar that can turn the status bar white over the
/// blue band (see `ChatTopBar`).
struct ChatScreen: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var router: AppRouter

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                EngineStatusView()
                Group {
                    if chat.conversation.isEmpty && !chat.isGenerating {
                        EmptyChatView()
                    } else {
                        MessageList()
                    }
                }
                .frame(maxHeight: .infinity)
                if chat.conversation.isEmpty && !chat.isGenerating && !isEngineFailed {
                    SuggestionChips()
                }
                ComposerView()
            }
            .background(Theme.background.ignoresSafeArea())
            .modifier(ChatTopBar())
        }
        .modifier(AttachmentSheetPresenter())
        .onChange(of: chat.phase) { old, new in
            playHaptic(from: old, to: new)
        }
    }

    private var isEngineFailed: Bool {
        if case .failed = chat.engineState { return true }
        return false
    }

    /// Voice mode has its own feedback, so the chat's haptics stay quiet
    /// while it is up even though its turns stream through here.
    private func playHaptic(from old: ChatController.ReplyPhase, to new: ChatController.ReplyPhase) {
        guard settings.haptics, !router.isVoiceModePresented else { return }
        switch (old, new) {
        case (.waiting, .thinking), (.waiting, .writing):
            ChatHaptics.replyStarted(enabled: true)
        case (_, .idle) where old != .idle:
            if chat.conversation.messages.last?.wasStopped != true {
                ChatHaptics.replyFinished(enabled: true)
            }
        default:
            break
        }
    }
}
