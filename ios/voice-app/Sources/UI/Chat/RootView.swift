import SwiftUI
import UIKit

/// The whole app: ChatGPT's left drawer over the chat screen, with
/// Settings and voice mode presented from the router.
///
/// The drawer is not a navigation container. The chat screen slides right
/// to reveal the sidebar and is dimmed while it is pushed aside, as in
/// ChatGPT's iPhone app; a tap on the dimmed chat or a swipe to the left
/// (on the chat or on the sidebar itself) closes it again, and a drag in
/// from the left edge opens it.
struct RootView: View {
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var router: AppRouter

    /// The finger's travel during a drawer drag on the edge strip or the
    /// scrim; zero at rest.
    @State private var dragOffset: CGFloat = 0
    /// A closing swipe on the sidebar. Gesture state rather than `@State`
    /// because the history list's scrolling can cancel the gesture, which
    /// skips `onEnded`; this resets either way, so the drawer never sticks
    /// half shut.
    @GestureState private var sidebarSwipe = SidebarSwipe()

    private static let sidebarFraction: CGFloat = 0.84
    private static let spring = Animation.spring(response: 0.36, dampingFraction: 0.86)
    /// Clears the top bar (44 points, 54 on iOS 26), which the edge-drag
    /// strip leaves alone so the sidebar button stays tappable.
    private static let topBarClearance: CGFloat = 56

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let sidebarWidth = (width * Self.sidebarFraction).rounded()
            let resting = router.isSidebarOpen ? sidebarWidth : 0
            let drag = dragOffset + sidebarSwipe.translation
            let offset = min(max(resting + drag, 0), sidebarWidth)
            let progress = sidebarWidth > 0 ? offset / sidebarWidth : 0
            // Follow the finger exactly while dragging; spring otherwise,
            // whoever opened or closed the drawer.
            let motion = drag == 0 ? Self.spring : nil

            ZStack(alignment: .topLeading) {
                SidebarView()
                    .frame(width: sidebarWidth)
                    .simultaneousGesture(sidebarCloseSwipe(sidebarWidth: sidebarWidth))
                    .animation(motion) { $0.offset(x: offset - sidebarWidth) }
                    .accessibilityHidden(!router.isSidebarOpen)
                    .accessibilityAction(.escape) { router.isSidebarOpen = false }

                ChatScreen()
                    .frame(width: width)
                    .overlay(alignment: .topLeading) { edgeDragStrip(sidebarWidth: sidebarWidth) }
                    .animation(motion) { $0.offset(x: offset) }
                    .accessibilityHidden(router.isSidebarOpen)

                // A sibling of the chat rather than an overlay on it: the
                // chat is hidden from VoiceOver while the sidebar is open,
                // and this is the sidebar's way out.
                scrim(progress: progress, width: width, offset: offset, sidebarWidth: sidebarWidth, motion: motion)
            }
        }
        .overlay(alignment: .top) { BannerHost() }
        .sheet(isPresented: $router.isSettingsPresented) { SettingsView() }
        .fullScreenCover(isPresented: $router.isVoiceModePresented) { VoiceModeView() }
        .onChange(of: router.isSidebarOpen) { _, _ in
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
        .tint(Theme.accent)
        .preferredColorScheme(settings.appearance.colorScheme)
    }

    // MARK: - Drawer

    /// Dims the part of the chat still showing beside the sidebar.
    @ViewBuilder
    private func scrim(progress: CGFloat, width: CGFloat, offset: CGFloat, sidebarWidth: CGFloat, motion: Animation?) -> some View {
        if progress > 0 {
            Rectangle()
                .fill(Theme.yale.opacity(0.32))
                .animation(motion) { $0.opacity(progress) }
                .frame(width: width)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture { router.isSidebarOpen = false }
                .gesture(closeDrag(sidebarWidth: sidebarWidth))
                .animation(motion) { $0.offset(x: offset) }
                .accessibilityElement()
                .accessibilityLabel("Close sidebar")
                .accessibilityAddTraits(.isButton)
                .accessibilityAction { router.isSidebarOpen = false }
        }
    }

    /// A thin strip along the left edge, below the top bar, that starts the
    /// drag to open. Kept to the edge so it never competes with scrolling
    /// code blocks, tables or the chips.
    @ViewBuilder
    private func edgeDragStrip(sidebarWidth: CGFloat) -> some View {
        if !router.isSidebarOpen {
            Color.clear
                .frame(width: 18)
                .frame(maxHeight: .infinity)
                .padding(.top, Self.topBarClearance)
                .contentShape(Rectangle())
                .gesture(openDrag(sidebarWidth: sidebarWidth))
                .accessibilityHidden(true)
        }
    }

    private func openDrag(sidebarWidth: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 8, coordinateSpace: .global)
            .onChanged { value in
                dragOffset = max(0, value.translation.width)
            }
            .onEnded { value in
                let opens = value.predictedEndTranslation.width > sidebarWidth * 0.5
                    || value.translation.width > sidebarWidth * 0.4
                settle(open: opens)
            }
    }

    private func closeDrag(sidebarWidth: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 8, coordinateSpace: .global)
            .onChanged { value in
                dragOffset = min(0, value.translation.width)
            }
            .onEnded { value in
                let closes = value.predictedEndTranslation.width < -sidebarWidth * 0.3
                    || value.translation.width < -sidebarWidth * 0.4
                settle(open: !closes)
            }
    }

    /// The sidebar's own leftward swipe. It runs alongside the history
    /// list's vertical scrolling, so the drawer only follows a drag whose
    /// first movement is mostly sideways and to the left.
    private func sidebarCloseSwipe(sidebarWidth: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 12, coordinateSpace: .global)
            .updating($sidebarSwipe) { value, swipe, _ in
                let dx = value.translation.width
                if swipe.isClosing == nil {
                    swipe.isClosing = dx < 0 && abs(dx) > abs(value.translation.height) * 1.5
                }
                if swipe.isClosing == true {
                    swipe.translation = min(0, dx)
                }
            }
            .onEnded { value in
                let dx = value.translation.width
                guard dx < 0, abs(dx) > abs(value.translation.height) else { return }
                let closes = value.predictedEndTranslation.width < -sidebarWidth * 0.3
                    || dx < -sidebarWidth * 0.4
                if closes { router.isSidebarOpen = false }
            }
    }

    /// Ends a drag: the drawer springs from wherever the finger left it.
    private func settle(open: Bool) {
        router.isSidebarOpen = open
        dragOffset = 0
    }
}

/// A swipe on the sidebar: whether it was judged a closing swipe (nil
/// until the finger has moved), and how far the drawer has followed it.
private struct SidebarSwipe {
    var isClosing: Bool?
    var translation: CGFloat = 0
}

#if DEBUG
/// A backend that answers at once with canned Markdown, so the preview
/// runs without the engine.
private final class PreviewBackend: ConversationBackend {
    var engineDescription: String { "Preview" }

    func warmUp() async -> String? { nil }

    func reply(
        _ ticket: ReplyTicket,
        session: String,
        messages: [PromptMessage],
        options: ReplyOptions,
        onEvent: @escaping (ReplyEvent) -> Void
    ) async throws -> ReplyResult {
        let text = "Here is **a reply** with `code` and a list:\n\n- one\n- two"
        onEvent(.text(text))
        return ReplyResult(text: text, reasoning: "", stats: TurnStats(), cancelled: false)
    }

    func cancel(_ ticket: ReplyTicket) {}
}

#Preview {
    let settings = AppSettings()
    let store = ChatStore()
    let speech = SpeechSynthesis()
    let microphone = MicrophoneInput()
    let chat = ChatController(store: store, backend: PreviewBackend(), speech: speech, settings: settings)
    let voice = VoiceModeController(chat: chat, microphone: microphone, sample: nil, speech: speech, settings: settings)
    return RootView()
        .environmentObject(settings)
        .environmentObject(store)
        .environmentObject(chat)
        .environmentObject(voice)
        .environmentObject(DictationController(microphone: microphone))
        .environmentObject(AppRouter())
        .environmentObject(VoicePreview(speech: speech, chat: chat))
}
#endif
