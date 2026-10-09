import SwiftUI

/// The band across the top, in the site's Yale blue with white content:
/// the sidebar button, the title menu, and either the temporary-chat
/// toggle (on an empty chat) or the new-chat button.
///
/// It is the chat's navigation bar rather than a plain view because only a
/// bar SwiftUI manages can set the status bar's style. Told that the band
/// is dark, the stack turns the clock and battery white; over a custom band
/// they stay black in light mode, about 1.7:1 against the blue. While the
/// sidebar is open its light top sits under the clock instead, so the bar
/// goes back to the app's own scheme. The style is not set app-wide
/// because voice mode and the sidebar have light tops in light mode.
struct ChatTopBar: ViewModifier {
    @EnvironmentObject private var chat: ChatController
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
                    trailingButton
                }
                .withoutGlassBackground()
            }
            .toolbarBackground(Theme.topBar, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarColorScheme(router.isSidebarOpen ? nil : .dark, for: .navigationBar)
    }

    private var sidebarButton: some View {
        Button {
            router.isSidebarOpen = true
        } label: {
            SidebarGlyph()
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(Theme.onTopBar)
        .accessibilityLabel("Open sidebar")
        .accessibilityShowsLargeContentViewer {
            Label("Open sidebar", systemImage: "sidebar.leading")
        }
    }

    @ViewBuilder
    private var trailingButton: some View {
        if chat.conversation.isEmpty {
            TemporaryChatToggle()
        } else {
            Button {
                chat.newChat()
            } label: {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 19, weight: .regular))
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(Theme.onTopBar)
            .accessibilityLabel("New chat")
            .accessibilityShowsLargeContentViewer {
                Label("New chat", systemImage: "square.and.pencil")
            }
        }
    }
}

/// Switches an empty chat between saved and temporary. Nothing has been
/// sent yet, so the switch is about the composer's contents: what was typed
/// carries across. Pending photos and files cannot (the controller only
/// takes them by importing them afresh), so the toggle waits until they
/// are sent or removed rather than silently dropping them.
private struct TemporaryChatToggle: View {
    @EnvironmentObject private var chat: ChatController

    var body: some View {
        let isTemporary = chat.conversation.isTemporary
        let isBlocked = !chat.pendingAttachments.isEmpty || chat.isImportingAttachment
        Button {
            let draft = chat.draft
            chat.newChat(temporary: !isTemporary)
            chat.draft = draft
        } label: {
            TemporaryChatGlyph(isFilled: isTemporary, size: 21)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(Theme.onTopBar)
        .opacity(isBlocked ? 0.45 : 1)
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
