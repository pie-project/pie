import SwiftUI

/// ChatGPT's sidebar: search and a new-chat button at the top, shortcuts
/// to a new, temporary or spoken chat, the saved conversations grouped by
/// recency, and the app's row at the bottom that opens Settings.
///
/// A conversation is renamed or deleted from its long-press menu, and
/// deleting asks first. The rows have no swipe actions: a leftward swipe
/// anywhere on the sidebar closes it, as in ChatGPT, and on a row it would
/// otherwise reveal (or, swiped far enough, carry out) a delete.
///
/// The sidebar is always there, off screen while shut, so it watches as
/// little as it can. Of the open chat it needs only the id (for the
/// highlight), and `SidebarContent` compares equal unless that changed: a
/// streamed token or a keystroke, which every observer of the chat hears
/// about, does not regroup or redraw the history.
struct SidebarView: View {
    @EnvironmentObject private var chat: ChatController

    var body: some View {
        SidebarContent(chat: chat, openID: chat.conversation.id)
            .equatable()
    }
}

private struct SidebarContent: View, Equatable {
    /// For its actions only; reading it here would not redraw this view.
    let chat: ChatController
    /// The open conversation, highlighted in the list.
    let openID: UUID

    @EnvironmentObject private var store: ChatStore
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var newChat: NewChatTransition

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var query = ""
    @State private var renaming: Conversation?
    @State private var renameText = ""
    @State private var deleting: Conversation?

    static func == (lhs: SidebarContent, rhs: SidebarContent) -> Bool {
        lhs.openID == rhs.openID
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            List {
                shortcuts
                history
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .scrollDismissesKeyboard(.immediately)
            .environment(\.defaultMinListRowHeight, 36)
            // Results and their section headers slide and fade into place
            // as the query changes, rather than blinking; with Reduce
            // Motion they only fade.
            .animation(Motion.reduced(Motion.content, reduceMotion), value: query)
            Rectangle().fill(Theme.hairline).frame(height: 0.5)
            settingsRow
        }
        .background(Theme.sidebar.ignoresSafeArea())
        .alert("Rename chat", isPresented: isPresenting($renaming), presenting: renaming) { conversation in
            TextField("Chat title", text: $renameText)
            Button("Cancel", role: .cancel) {}
            Button("Rename") { rename(conversation) }
        }
        .alert("Delete chat?", isPresented: isPresenting($deleting), presenting: deleting) { conversation in
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) { delete(conversation) }
        } message: { conversation in
            Text("This will delete \u{201C}\(conversation.displayTitle)\u{201D}.")
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(Theme.secondaryInk)
                    .accessibilityHidden(true)
                TextField("Search", text: $query)
                    .foregroundStyle(Theme.ink)
                    .autocorrectionDisabled()
                    .submitLabel(.search)
                if !query.isEmpty {
                    Button {
                        query = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(Theme.tertiaryInk)
                    }
                    .buttonStyle(PressDimButtonStyle())
                    .accessibilityLabel("Clear search")
                    .transition(.popIn())
                }
            }
            // The clear button pops in with the first letter and out when
            // the field is emptied.
            .animation(Motion.control, value: query.isEmpty)
            .padding(.horizontal, 12)
            .frame(minHeight: 40)
            .background(Theme.surfaceStrong.opacity(0.6), in: RoundedRectangle(cornerRadius: 12, style: .continuous))

            Button {
                startChat(temporary: false)
            } label: {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 19))
                    .foregroundStyle(Theme.ink)
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(PressDimButtonStyle())
            .accessibilityLabel("New chat")
        }
        .padding(.leading, 14)
        .padding(.trailing, 6)
        .padding(.vertical, 6)
    }

    private var shortcuts: some View {
        Group {
            SidebarShortcutRow(title: "New chat") {
                Image(systemName: "square.and.pencil")
            } action: {
                startChat(temporary: false)
            }
            SidebarShortcutRow(title: "Temporary chat") {
                TemporaryChatGlyph(size: 19)
            } action: {
                startChat(temporary: true)
            }
            SidebarShortcutRow(title: "Talk to Pie") {
                Image(systemName: "waveform")
            } action: {
                // Voice mode fades in once the drawer has slid shut, so
                // the two motions do not cross.
                KeyboardDismissal.dismiss()
                router.closeSidebar { router.isVoiceModePresented = true }
            }
        }
        .listRowBackground(Color.clear)
        .listRowSeparator(.hidden)
        .listRowInsets(Self.rowInsets)
    }

    @ViewBuilder
    private var history: some View {
        let sections = (query.isEmpty ? store.groupedHistory() : ChatStore.grouped(store.search(query)))
            .map { HistorySection(title: $0.title, items: $0.items) }
        if sections.isEmpty && !query.isEmpty {
            Text("No chats match \u{201C}\(query)\u{201D}")
                .font(.subheadline)
                .foregroundStyle(Theme.secondaryInk)
                .padding(.top, 16)
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)
                .listRowInsets(Self.headerInsets)
        }
        ForEach(sections) { section in
            Text(section.title)
                .font(.footnote.weight(.semibold))
                .foregroundStyle(Theme.secondaryInk)
                .padding(.top, 18)
                .padding(.bottom, 2)
                .accessibilityAddTraits(.isHeader)
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)
                .listRowInsets(Self.headerInsets)
            ForEach(section.items) { conversation in
                conversationRow(conversation)
            }
        }
    }

    private func conversationRow(_ conversation: Conversation) -> some View {
        SidebarConversationRow(conversation: conversation, isOpen: conversation.id == openID) {
            open(conversation)
        }
        .contextMenu {
            Button {
                renameText = conversation.title.isEmpty ? conversation.displayTitle : conversation.title
                renaming = conversation
            } label: {
                Label("Rename", systemImage: "pencil")
            }
            Button(role: .destructive) {
                deleting = conversation
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
        .listRowBackground(Color.clear)
        .listRowSeparator(.hidden)
        .listRowInsets(Self.rowInsets)
    }

    private var settingsRow: some View {
        Button {
            router.isSettingsPresented = true
        } label: {
            HStack(spacing: 12) {
                Text("P")
                    .font(Theme.serif(17))
                    .foregroundStyle(Theme.onTopBar)
                    .frame(width: 34, height: 34)
                    .background(Theme.yale, in: Circle())
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Pie Voice")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.ink)
                    Text("On this iPhone \u{00B7} \(PieRuntimeConfig.pieVersion)")
                        .font(.caption)
                        .foregroundStyle(Theme.secondaryInk)
                }
                Spacer(minLength: 0)
                Image(systemName: "gearshape")
                    .foregroundStyle(Theme.secondaryInk)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressHighlightButtonStyle(cornerRadius: 12))
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .accessibilityLabel("Pie Voice settings")
    }

    // MARK: - Actions

    /// ChatGPT swaps the chat behind the sidebar at once (no fade, no
    /// scrolling) and lets the closing drawer reveal it, already at its
    /// last message. The keyboard goes first (`setSidebarOpen`), so the
    /// layout settles before anything else changes.
    private func open(_ conversation: Conversation) {
        router.setSidebarOpen(false)
        var swap = Transaction()
        swap.disablesAnimations = true
        withTransaction(swap) {
            chat.open(conversation.id)
        }
    }

    /// The chat behind fades to the greeting as the drawer closes over it.
    /// The keyboard goes first; then the drawer's flag and the fade are
    /// one transaction, since an unanimated published change made in the
    /// same moment as an animated one can cancel it.
    private func startChat(temporary: Bool) {
        KeyboardDismissal.dismiss()
        withMotion(Motion.crossfade) {
            router.isSidebarOpen = false
            newChat.start(chat, temporary: temporary)
        }
    }

    /// The store animates the row's new title in place.
    private func rename(_ conversation: Conversation) {
        let title = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { return }
        store.rename(conversation.id, to: title)
    }

    /// The row collapses out of the list (the store animates it).
    private func delete(_ conversation: Conversation) {
        guard conversation.id == chat.conversation.id else {
            store.delete(conversation.id)
            return
        }
        // The open chat: start a fresh one instead, as ChatGPT does (the
        // controller would otherwise save the deleted thread again on its
        // next change). The sidebar stays open, so the chat beside it
        // visibly fades to the greeting; the delete and the swap happen
        // together once it has faded out.
        newChat.start(chat) {
            store.delete(conversation.id)
        }
    }

    private func isPresenting(_ item: Binding<Conversation?>) -> Binding<Bool> {
        Binding(get: { item.wrappedValue != nil }, set: { if !$0 { item.wrappedValue = nil } })
    }

    /// Rows run nearly edge to edge so their pressed and open highlights
    /// are inset 8 points, as in ChatGPT; their text keeps the 20-point
    /// margin of the section titles.
    private static let rowInsets = EdgeInsets(top: 0, leading: 8, bottom: 0, trailing: 8)
    private static let headerInsets = EdgeInsets(top: 0, leading: 20, bottom: 0, trailing: 16)
}

private struct HistorySection: Identifiable {
    let title: String
    let items: [Conversation]

    var id: String { title }
}
