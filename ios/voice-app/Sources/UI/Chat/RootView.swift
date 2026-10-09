import SwiftUI
import UIKit

/// The whole app: ChatGPT's left drawer over the chat screen, with
/// Settings and voice mode presented from the router.
///
/// The drawer is not a navigation container. The chat screen slides right
/// to reveal the sidebar and is dimmed while it is pushed aside, as in
/// ChatGPT's iPhone app. A sideways drag almost anywhere on the chat opens
/// it; a tap on the dimmed chat, or a drag to the left on the chat or the
/// sidebar, closes it again.
struct RootView: View {
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var router: AppRouter

    /// Shared by the top bar's and the sidebar's new-chat buttons and the
    /// chat screen that fades for them.
    @StateObject private var newChat = NewChatTransition()

    var body: some View {
        SidebarDrawer()
            .sheet(isPresented: $router.isSettingsPresented) { SettingsView() }
            .fullScreenCover(isPresented: $router.isVoiceModePresented) { VoiceModeView() }
            .onChange(of: router.isVoiceModePresented) { _, presented in
                Haptics.isStreamingSuppressed = presented
            }
            .tint(Theme.accent)
            .preferredColorScheme(settings.appearance.colorScheme)
            .environmentObject(newChat)
    }
}

/// The sidebar and the chat side by side, moved together by one position
/// (`SidebarDrawerModel`), with the drag that moves them.
///
/// `router.isSidebarOpen` says where the drawer is going; the model says
/// where it is. Whoever flips the flag (the top bar's button, a sidebar
/// row, the tour) gets the slide; a drag moves the drawer itself and sets
/// the flag when the finger lets go.
private struct SidebarDrawer: View {
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Held, not observed (`@State`, not `@StateObject`): only
    /// `DrawerLayers` observes the position, so only it redraws on each
    /// frame of a slide.
    @State private var drawer = SidebarDrawerModel()

    /// iOS 17's stand-in for the UIKit pan: whether the current drag was
    /// judged the drawer's (nil until the finger has moved), and where the
    /// finger was then.
    @State private var fallbackClaim: Bool?
    @State private var fallbackStart: CGFloat = 0
    @GestureState private var fallbackIsActive = false

    private static let sidebarFraction: CGFloat = 0.84

    var body: some View {
        GeometryReader { proxy in
            let sidebarWidth = (proxy.size.width * Self.sidebarFraction).rounded()
            let layers = DrawerLayers(drawer: drawer, width: proxy.size.width, sidebarWidth: sidebarWidth)
            if #available(iOS 18.0, *) {
                layers.gesture(SidebarPanGesture(
                    position: { [drawer] in drawer.position },
                    onBegan: dragBegan,
                    onChanged: { [drawer] in drawer.drag(by: $0, width: sidebarWidth) },
                    onEnded: { dragEnded(velocity: $0, sidebarWidth: sidebarWidth) }
                ))
            } else {
                layers.simultaneousGesture(fallbackDrag(sidebarWidth: sidebarWidth))
            }
        }
        .onAppear {
            drawer.onClosed = { [router] in router.sidebarDidClose() }
        }
        .onChange(of: router.isSidebarOpen) { _, open in
            // ChatGPT drops the keyboard as the drawer moves (the
            // composer's, or the sidebar search's).
            Self.dismissKeyboard()
            drawer.move(toOpen: open, reduceMotion: reduceMotion)
        }
        .onChange(of: fallbackIsActive) { _, isActive in
            // The system can cancel a SwiftUI drag without calling
            // `onEnded`; settle the drawer then rather than leave it half
            // open.
            guard !isActive, fallbackClaim != nil else { return }
            if fallbackClaim == true { dragEnded(velocity: 0, sidebarWidth: 0) }
            fallbackClaim = nil
        }
    }

    // MARK: - Dragging

    private func dragBegan() {
        Self.dismissKeyboard()
        drawer.beginDrag()
    }

    private func dragEnded(velocity: CGFloat, sidebarWidth: CGFloat) {
        let wasOpen = router.isSidebarOpen
        let opens = drawer.endDrag(velocity: velocity, width: sidebarWidth, reduceMotion: reduceMotion)
        if opens != wasOpen { router.isSidebarOpen = opens }
        // A light tick as a drag lets the drawer snap open; none for the
        // button, none for closing.
        if opens && !wasOpen { Haptics.selection(enabled: settings.haptics) }
    }

    /// iOS 17 has no way to put a UIKit recognizer on a SwiftUI view, so it
    /// gets a SwiftUI drag that runs alongside scrolling and claims the
    /// touch only if its first movement is clearly sideways, in a direction
    /// the drawer can go. (Unlike the UIKit pan, it cannot tell a sideways
    /// scroller from the chat, so a code block dragged right on iOS 17 also
    /// opens the drawer.)
    private func fallbackDrag(sidebarWidth: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 10, coordinateSpace: .global)
            .updating($fallbackIsActive) { _, isActive, _ in isActive = true }
            .onChanged { value in
                let dx = value.translation.width
                if fallbackClaim == nil {
                    let position = drawer.position
                    let isSideways = abs(dx) > abs(value.translation.height) * 1.2
                    let canGo = dx > 0 ? position < 1 : position > 0
                    fallbackClaim = isSideways && canGo
                    if fallbackClaim == true {
                        fallbackStart = dx
                        dragBegan()
                    }
                }
                if fallbackClaim == true {
                    drawer.drag(by: dx - fallbackStart, width: sidebarWidth)
                }
            }
            .onEnded { value in
                if fallbackClaim == true {
                    dragEnded(velocity: value.velocity.width, sidebarWidth: sidebarWidth)
                }
                fallbackClaim = nil
            }
    }

    private static func dismissKeyboard() {
        KeyboardDismissal.dismiss()
    }
}

/// The sidebar, the chat and the dim over the chat, placed by the
/// drawer's live position. This is the only view that reads the position,
/// so it is the only one that redraws on each frame of a slide; the
/// sidebar and the chat inside it are not redrawn, only moved.
private struct DrawerLayers: View {
    @ObservedObject var drawer: SidebarDrawerModel
    let width: CGFloat
    let sidebarWidth: CGFloat

    @EnvironmentObject private var router: AppRouter

    var body: some View {
        let progress = drawer.position
        let offset = progress * sidebarWidth

        ZStack(alignment: .topLeading) {
            SidebarView()
                .frame(width: sidebarWidth)
                .offset(x: offset - sidebarWidth)
                .accessibilityHidden(!router.isSidebarOpen)
                .accessibilityAction(.escape) { router.isSidebarOpen = false }

            ChatScreen()
                .frame(width: width)
                // The status bar turns dark over the sidebar's light top
                // once the sidebar covers most of it, not when a button
                // is tapped or a finger lets go.
                .environment(\.sidebarCoversStatusBar, progress > 0.5)
                .offset(x: offset)
                .accessibilityHidden(router.isSidebarOpen)

            // A sibling of the chat rather than an overlay on it: the
            // chat is hidden from VoiceOver while the sidebar is open,
            // and this is the sidebar's way out.
            scrim(progress: progress)
                .offset(x: offset)
        }
        // Behind both, in case a fraction of a point ever opens between
        // them.
        .background(Theme.sidebar.ignoresSafeArea())
    }

    /// Dims the part of the chat still showing beside the sidebar. Always
    /// there, its strength following the drawer, so it fades in and out
    /// with the slide rather than popping; it takes touches only while the
    /// sidebar is open.
    private func scrim(progress: CGFloat) -> some View {
        Rectangle()
            .fill(Theme.yale.opacity(0.32))
            .opacity(progress)
            .frame(width: width)
            .ignoresSafeArea()
            .contentShape(Rectangle())
            .onTapGesture { router.isSidebarOpen = false }
            .allowsHitTesting(router.isSidebarOpen)
            .accessibilityElement()
            .accessibilityLabel("Close sidebar")
            .accessibilityAddTraits(.isButton)
            .accessibilityAction { router.isSidebarOpen = false }
            .accessibilityHidden(!router.isSidebarOpen)
    }
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
