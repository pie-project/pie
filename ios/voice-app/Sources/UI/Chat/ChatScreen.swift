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
                ComposerSlot()
            }
            .overlay(alignment: .top) { BannerHost() }
            .background(Theme.background.ignoresSafeArea())
            .modifier(ChatTopBar())
        }
        .modifier(AttachmentSheetPresenter())
    }
}

/// The composer, which takes no touches during a new chat's crossfade: a
/// message sent then would land in the conversation being left, and be
/// cut off by the swap (`NewChatTransition`).
private struct ComposerSlot: View {
    @EnvironmentObject private var newChat: NewChatTransition

    var body: some View {
        ComposerView()
            .allowsHitTesting(!newChat.isCrossfading)
    }
}

/// Everything above the composer: the engine's status, then either the
/// greeting and the starter chips or the conversation.
///
/// Its changes animate here, whoever made them. The greeting and the chips
/// fade out as the first message arrives, the transcript rising a few
/// points as it fades in; a new chat crossfades from the transcript to the
/// greeting (`NewChatTransition`). The boot pill leaving eases the content
/// under it rather than jumping it. The composer is outside all of this,
/// so it never moves with it.
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
        // During a new chat's crossfade the greeting is already fading in
        // over the transcript of the conversation being left.
        let showsGreeting = showsEmptyState || newChat.isCrossfading
        VStack(spacing: 0) {
            EngineStatusView()
            // A ZStack, so that while one fades into the other both share
            // the whole space rather than splitting it between them. The
            // reader hands the greeting the space's height for its exit
            // (`PinnedFade`).
            GeometryReader { space in
                ZStack {
                    // The greeting in a container of its own, so that its
                    // exit gets its own short fade without changing how the
                    // transcript arrives.
                    ZStack {
                        if showsGreeting {
                            greeting
                                // Arrives with the caller's fade (a new
                                // chat's crossfade); leaves in place.
                                .transition(.asymmetric(
                                    insertion: .opacity,
                                    removal: AnyTransition(PinnedFade(height: space.size.height))
                                ))
                        }
                    }
                    .transaction(value: showsGreeting) { transaction in
                        // Leaving, it fades fast, as ChatGPT's greeting does
                        // when the first bubble arrives. An animation
                        // attached to the transition itself lost to the
                        // send's longer one, and the greeting lingered for
                        // half a second over the new transcript (recorded).
                        guard !showsGreeting, !transaction.disablesAnimations else { return }
                        transaction.animation = Motion.fadeOut
                    }
                    if !showsEmptyState {
                        MessageList()
                            // Faded out under the greeting for a new chat,
                            // it is then removed at once rather than faded
                            // again.
                            .opacity(newChat.isCrossfading ? 0 : 1)
                            .transition(newChat.isCrossfading ? .identity : .fadeRise(12))
                    }
                }
                .frame(width: space.size.width, height: space.size.height)
            }
        }
        // Nothing here takes a touch while a new chat crossfades in: a chip
        // or a tap on the old transcript would act on the chat being left.
        .allowsHitTesting(!newChat.isCrossfading)
        // Innermost first: the last of these sees the change first, so the
        // empty state's own animation wins when both change together.
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

    /// The greeting and the chips under it, one layer, so that the
    /// transcript keeps the whole space while they fade in over it: chips
    /// taking their room from it would shift the outgoing transcript up
    /// mid-fade.
    private var greeting: some View {
        VStack(spacing: 0) {
            EmptyChatView()
            if !isEngineFailed {
                SuggestionChips()
                    .transition(.asymmetric(
                        insertion: .opacity,
                        removal: .opacity.animation(Motion.fadeOut)
                    ))
            }
        }
    }

    /// Gives an unanimated change `animation`, leaving alone a change that
    /// already has one or that asked for none.
    private static func fillIn(_ transaction: inout Transaction, with animation: Animation) {
        guard transaction.animation == nil, !transaction.disablesAnimations else { return }
        transaction.animation = Motion.reduced(animation)
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

/// The greeting and the chips leaving with the first message: they fade
/// where they are.
///
/// The greeting is centred in the space above the composer, and that space
/// grows as the keyboard goes down with the send. When the keyboard's
/// layout change and the send land in the same update, the greeting,
/// laid out as usual while it faded, slid about 130 pt down the screen
/// (recorded). So on its way out the layer keeps the height of the space
/// it was last shown in, pinned to the top: the space can grow under it
/// without moving it. (That height comes from the layout pass that last
/// showed it. Measured afterwards and stored, it was a step behind when
/// the keyboard had gone in an earlier update, and the greeting slid up
/// instead, recorded.)
private struct PinnedFade: Transition {
    /// The greeting area's height while the greeting showed; nil before it
    /// was measured, when there is nothing to pin.
    let height: CGFloat?

    func body(content: Content, phase: TransitionPhase) -> some View {
        content
            .frame(height: phase == .didDisappear ? height : nil)
            .frame(maxHeight: .infinity, alignment: .top)
            .opacity(phase.isIdentity ? 1 : 0)
    }
}
