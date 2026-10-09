import SwiftUI

/// The chat: the top bar, the engine's status, the conversation (or the
/// empty state and starter chips), the composer pinned above the keyboard,
/// and the toast under the bar.
///
/// The navigation stack never pushes anything. It is there for its bar,
/// the only kind of top bar that can turn the status bar white over the
/// blue band (see `ChatTopBar`).
///
/// This view reads nothing from the chat; `ChatContent` and the bar's own
/// items do. So a streamed token or a keystroke redraws those pieces, not
/// the navigation stack and its bar.
struct ChatScreen: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ChatContent()
                ComposerView()
            }
            .overlay(alignment: .top) { BannerHost() }
            .background(Theme.background.ignoresSafeArea())
            .modifier(ChatTopBar())
        }
        .modifier(AttachmentSheetPresenter())
    }
}

/// Everything above the composer: the engine's status, then either the
/// greeting and the starter chips or the conversation.
///
/// Its changes animate here, whoever made them. The greeting and the chips
/// fade out as the first message arrives, the transcript rising a few
/// points as it fades in; a new chat fades the transcript out and the
/// greeting back in (`NewChatTransition`). Layout shifts
/// (the boot pill leaving, the chips making way while the user types)
/// ease rather than jump. The composer is outside all of this, so it never
/// moves with it.
///
/// Each of these fills in an animation only when the change came without
/// one. A caller that chose its own keeps it: a temporary chat's
/// crossfade, a send's motion, and opening a saved chat from the sidebar,
/// which ChatGPT swaps with no animation at all under the closing drawer.
private struct ChatContent: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var newChat: NewChatTransition

    /// The boot pill's own animation (`EngineStatusView`), so the content
    /// below it moves in step as it leaves.
    private static let engineStatusAnimation = Animation.easeInOut(duration: 0.25)

    var body: some View {
        let showsEmptyState = chat.conversation.isEmpty && !chat.isGenerating
        VStack(spacing: 0) {
            EngineStatusView()
            // A ZStack, so that while one fades into the other both share
            // the whole space rather than splitting it between them.
            ZStack {
                if showsEmptyState {
                    EmptyChatView()
                        // Leaves fast, as ChatGPT's greeting does when the
                        // first bubble arrives: lingering, it showed through
                        // the new transcript for half a second (recorded).
                        .transition(.asymmetric(
                            insertion: .opacity,
                            removal: .opacity.animation(Motion.fadeOut)
                        ))
                } else {
                    MessageList()
                        // Faded out for a new chat, it is then removed
                        // at once rather than faded a second time.
                        .opacity(newChat.isTranscriptFadedOut ? 0 : 1)
                        .transition(newChat.isTranscriptFadedOut ? .identity : .fadeRise(12))
                }
            }
            .frame(maxHeight: .infinity)
            if showsEmptyState && !isEngineFailed {
                SuggestionChips()
                    .transition(.asymmetric(
                        insertion: .opacity,
                        removal: .opacity.animation(Motion.fadeOut)
                    ))
            }
        }
        // Innermost first: the last of these sees the change first, so the
        // empty state's own animation wins when several change together.
        .transaction(value: chipsWanted) { transaction in
            // Only the empty state has chips; in a conversation a keystroke
            // must not lend an animation to anything else.
            guard showsEmptyState else { return }
            Self.fillIn(&transaction, with: chipsWanted ? Motion.fadeIn : Motion.fadeOut)
        }
        .transaction(value: engineStage) { transaction in
            Self.fillIn(&transaction, with: Self.engineStatusAnimation)
        }
        .transaction(value: showsEmptyState) { transaction in
            Self.fillIn(&transaction, with: Motion.content)
        }
        .onChange(of: chat.phase) { _, phase in
            // The reply's only haptic is the train of soft ticks as its
            // words appear (`Haptics.streamTick`, played by the
            // transcript), with nothing at the end, as in ChatGPT. Readying
            // the generator as the message goes out puts the first tick on
            // time. Voice mode has its own feedback.
            if phase == .waiting, !router.isVoiceModePresented {
                Haptics.prepareStreaming(enabled: settings.haptics)
            }
        }
    }

    /// Gives an unanimated change `animation`, leaving alone a change that
    /// already has one or that asked for none.
    private static func fillIn(_ transaction: inout Transaction, with animation: Animation) {
        guard transaction.animation == nil, !transaction.disablesAnimations else { return }
        transaction.animation = Motion.reduced(animation)
    }

    /// What `SuggestionChips` shows itself for: an empty composer.
    private var chipsWanted: Bool {
        chat.draft.isEmpty && chat.pendingAttachments.isEmpty && !chat.isImportingAttachment
    }

    /// The engine's state without the boot pill's seconds, which tick once
    /// a second and must not animate the screen each time.
    private var engineStage: EngineStage {
        switch chat.engineState {
        case .booting: return .booting
        case .ready: return .ready
        case .failed: return .failed
        }
    }

    private var isEngineFailed: Bool {
        engineStage == .failed
    }

    private enum EngineStage {
        case booting, ready, failed
    }
}
