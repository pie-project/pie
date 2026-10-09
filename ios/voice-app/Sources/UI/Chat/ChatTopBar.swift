import SwiftUI

/// The band across the top, in the site's Yale blue with white content:
/// the sidebar button, the title menu, and either the temporary-chat
/// toggle (on an empty chat) or the new-chat button.
///
/// It is the chat's navigation bar rather than a plain view because only a
/// bar SwiftUI manages can set the status bar's style. Told that the band
/// is dark, the stack turns the clock and battery white; over a custom band
/// they stay black in light mode, about 1.7:1 against the blue.
///
/// The bar's style never changes. Over the open sidebar and voice mode,
/// whose tops are light, the status bar is turned dark by
/// `StatusBarOverride` instead: switching this bar's color scheme there
/// restyled the whole bar mid-slide and stalled the drawer at halfway.
///
/// It reads nothing from the chat itself: the items that do (the title,
/// the trailing button) are their own views, so a streamed token or a
/// keystroke does not rebuild the bar.
struct ChatTopBar: ViewModifier {
    @EnvironmentObject private var router: AppRouter

    func body(content: Content) -> some View {
        content
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    sidebarButton
                }
                .withoutGlassBackground()
                ToolbarItem(placement: .principal) {
                    ChatTitleMenu()
                }
                ToolbarItem(placement: .topBarTrailing) {
                    TopBarTrailingButton()
                }
                .withoutGlassBackground()
            }
            .toolbarBackground(Theme.topBar, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
    }

    private var sidebarButton: some View {
        Button {
            // The keyboard drops and the drawer slides to match (see
            // `RootView`); no haptic, as in ChatGPT.
            router.setSidebarOpen(true)
        } label: {
            SidebarGlyph()
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(PressDimButtonStyle())
        .foregroundStyle(Theme.onTopBar)
        .accessibilityLabel("Open sidebar")
        .accessibilityShowsLargeContentViewer {
            Label("Open sidebar", systemImage: "sidebar.leading")
        }
    }
}

/// The bar's right-hand button: the temporary-chat toggle while the chat
/// is empty, the new-chat button once it is not. One fades into the other.
private struct TopBarTrailingButton: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var newChat: NewChatTransition

    var body: some View {
        let isEmpty = chat.conversation.isEmpty
        ZStack {
            if isEmpty {
                TemporaryChatToggle()
                    .transition(.opacity)
            } else {
                newChatButton
                    .transition(.opacity)
            }
        }
        .animation(Motion.crossfade, value: isEmpty)
    }

    private var newChatButton: some View {
        Button {
            // The conversation fades out and the greeting fades in, as in
            // ChatGPT; nothing slides.
            newChat.start(chat)
        } label: {
            Image(systemName: "square.and.pencil")
                .font(.system(size: 19, weight: .regular))
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(PressDimButtonStyle())
        .foregroundStyle(Theme.onTopBar)
        .accessibilityLabel("New chat")
        .accessibilityShowsLargeContentViewer {
            Label("New chat", systemImage: "square.and.pencil")
        }
    }
}

/// Switches an empty chat between saved and temporary. Nothing has been
/// sent yet, so the switch is about the composer's contents: what was typed
/// carries across, and so does the mode picked in the title menu. Pending
/// photos and files cannot (the controller only takes them by importing
/// them afresh), so the toggle waits until they are sent or removed rather
/// than silently dropping them.
private struct TemporaryChatToggle: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var settings: AppSettings

    var body: some View {
        let isTemporary = chat.conversation.isTemporary
        let isBlocked = !chat.pendingAttachments.isEmpty || chat.isImportingAttachment
        Button {
            let draft = chat.draft
            let mode = chat.mode
            // The glyph fills (or empties) and the greeting crossfades to
            // "Temporary Chat" (or back), with a selection tick.
            withMotion(Motion.crossfade) {
                chat.newChat(temporary: !isTemporary)
                chat.draft = draft
                chat.mode = mode
            }
            Haptics.selection(enabled: settings.haptics)
        } label: {
            TemporaryChatGlyph(isFilled: isTemporary, size: 21)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(PressDimButtonStyle())
        .foregroundStyle(Theme.onTopBar)
        .opacity(isBlocked ? 0.45 : 1)
        // An import flips this on and off within moments; easing it keeps
        // the button from blinking.
        .animation(Motion.control, value: isBlocked)
        .disabled(isBlocked)
        .accessibilityLabel("Temporary chat")
        .accessibilityValue(isTemporary ? "On" : "Off")
        .accessibilityHint(isBlocked ? "Available once the attached items are sent or removed" : "")
        .accessibilityShowsLargeContentViewer {
            Label("Temporary chat", systemImage: "bubble.left")
        }
    }
}

private extension ToolbarContent {
    /// On iOS 26 every bar button sits in its own glass capsule; on the
    /// site's flat blue band the buttons are plain white glyphs, as they
    /// are on earlier systems.
    func withoutGlassBackground() -> some ToolbarContent {
        if #available(iOS 26.0, *) {
            return sharedBackgroundVisibility(.hidden)
        }
        return self
    }
}
