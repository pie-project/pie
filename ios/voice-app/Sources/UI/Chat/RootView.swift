import SwiftUI
import UIKit

/// The whole app: ChatGPT's left drawer over the chat screen, with
/// Settings presented from the router and voice mode drawn over both.
///
/// The drawer is not a navigation container. The chat screen slides right
/// to reveal the sidebar and is dimmed while it is pushed aside, as in
/// ChatGPT's iPhone app. A sideways drag almost anywhere on the chat opens
/// it; a tap on the dimmed chat, or a drag to the left on the chat or the
/// sidebar, closes it again.
///
/// Voice mode is a layer over everything rather than a full-screen cover:
/// a cover can only slide up from the bottom, and ChatGPT's separate
/// voice mode fades in over the chat instead (about 0.35 s), its orb
/// growing in the middle. The router animates every change of
/// `isVoiceModePresented`, so the layer fades in and out whoever opens or
/// closes it.
struct RootView: View {
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var router: AppRouter

    /// Shared by the top bar's and the sidebar's new-chat buttons and the
    /// chat screen that fades for them.
    @StateObject private var newChat = NewChatTransition()

    var body: some View {
        ZStack {
            SidebarDrawer()
            if router.isVoiceModePresented {
                VoiceModeView()
                    // Its controls stay put if a keyboard was still on its
                    // way down when it arrived.
                    .ignoresSafeArea(.keyboard)
                    .transition(.opacity)
                    // Above the chat while it fades out, too.
                    .zIndex(1)
                    // VoiceOver stays inside voice mode, as it did in a
                    // full-screen cover. (Not `.accessibilityHidden` on the
                    // chat behind: `.accessibilityHidden(false)` there
                    // un-hid the sidebar and the dim inside it, and their
                    // elements covered the chat's buttons.)
                    .accessibilityAddTraits(.isModal)
            }
        }
        .sheet(isPresented: $router.isSettingsPresented) { SettingsView() }
        .onChange(of: router.isVoiceModePresented) { _, presented in
            // ChatGPT's voice mode does not tick as its replies stream.
            Haptics.isStreamingSuppressed = presented
            // The callers drop the keyboard before opening voice mode;
            // this catches one that did not, so it is not left up over it.
            // After this update, so the fade that has just started is not
            // laid out again in the middle of it.
            if presented {
                DispatchQueue.main.async { KeyboardDismissal.dismiss() }
            }
            // What a cover's presentation did on its own: VoiceOver moves
            // to the new screen.
            UIAccessibility.post(notification: .screenChanged, argument: nil)
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
            // The status bar's clock: in the left corner beside the Dynamic
            // Island or notch, about a fifth of the way across; in the
            // middle on iPhones with a Home button (a 20-point status bar).
            let clockX = proxy.size.width * (proxy.safeAreaInsets.top > 24 ? 0.18 : 0.5)
            let layers = DrawerLayers(drawer: drawer, width: proxy.size.width, sidebarWidth: sidebarWidth, clockX: clockX)
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
            // Taps drop the keyboard themselves, before the flag changes
            // (`AppRouter.setSidebarOpen`). This catches anything that set
            // the flag directly (the tour), so the keyboard never stays up
            // beside the open sidebar; from here the layout follows it a
            // little late.
            KeyboardDismissal.dismiss()
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

    /// Called by the pan itself, outside any SwiftUI update, so dropping
    /// the keyboard here lands the layout with it.
    private func dragBegan() {
        KeyboardDismissal.dismiss()
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

}

/// The sidebar, the chat and the dim over the chat, placed by the
/// drawer's live position. This is the only view that reads the position,
/// so it is the only one that redraws on each frame of a slide; the
/// sidebar and the chat inside it are not redrawn, only moved.
private struct DrawerLayers: View {
    @ObservedObject var drawer: SidebarDrawerModel
    let width: CGFloat
    let sidebarWidth: CGFloat
    /// Where the middle of the status bar's clock is, from the left edge.
    let clockX: CGFloat

    @EnvironmentObject private var router: AppRouter
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        let progress = drawer.position
        let offset = progress * sidebarWidth

        ZStack(alignment: .topLeading) {
            SidebarView()
                // The keyboard of the sidebar's search slides over the
                // bottom of the list, as in ChatGPT, instead of shrinking
                // the sidebar: shrunk, the list dropped its last rows in
                // one frame before the keyboard had even started to rise,
                // and the footer rode up after it (recorded).
                .ignoresSafeArea(.keyboard, edges: .bottom)
                .frame(width: sidebarWidth)
                .offset(x: offset - sidebarWidth)
                .accessibilityHidden(!router.isSidebarOpen)
                .accessibilityAction(.escape) { router.setSidebarOpen(false) }

            ChatScreen()
                .frame(width: width)
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
        .onChange(of: statusBarStyle(offset: offset), initial: true) { _, style in
            StatusBarOverride.apply(style)
        }
    }

    /// The status bar's clock and battery: white over the blue top bar,
    /// dark once the sidebar's light top has slid under the clock, and
    /// dark over voice mode. It changes as the sidebar's edge passes the
    /// clock, during a drag too, without touching the chat's navigation
    /// bar (`StatusBarOverride`). (Switched at halfway instead, the white
    /// clock sat invisible on the sidebar for a third of every slide.) In
    /// dark mode the sidebar and voice mode are dark as well, and nothing
    /// needs to change. A sheet over the sidebar (Settings) brings its own.
    private func statusBarStyle(offset: CGFloat) -> StatusBarOverride.Style {
        guard colorScheme == .light, !router.isSettingsPresented else { return .app }
        if router.isVoiceModePresented || offset > clockX { return .dark }
        return offset > 0 ? .light : .app
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
            .onTapGesture { router.setSidebarOpen(false) }
            .allowsHitTesting(router.isSidebarOpen)
            .accessibilityElement()
            .accessibilityLabel("Close sidebar")
            .accessibilityAddTraits(.isButton)
            .accessibilityAction { router.setSidebarOpen(false) }
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
