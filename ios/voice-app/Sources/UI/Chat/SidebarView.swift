import SwiftUI

/// ChatGPT's sidebar: search and a new-chat button at the top, shortcuts
/// to a new, temporary or spoken chat, the saved conversations grouped by
/// recency, and the app's row at the bottom that opens Settings.
///
/// A conversation is renamed or deleted from its long-press menu, and
/// deleting asks first. The rows have no swipe actions: a leftward swipe
/// anywhere on the sidebar closes it, as in ChatGPT, and on a row it would
/// otherwise reveal (or, swiped far enough, carry out) a delete.
struct SidebarView: View {
    @EnvironmentObject private var store: ChatStore
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var router: AppRouter

    @State private var query = ""
    @State private var renaming: Conversation?
    @State private var renameText = ""
    @State private var deleting: Conversation?

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
                    .buttonStyle(.plain)
                    .accessibilityLabel("Clear search")
                }
            }
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
            .buttonStyle(.plain)
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
                router.isSidebarOpen = false
                router.isVoiceModePresented = true
            }
        }
        .listRowBackground(Color.clear)
        .listRowSeparator(.hidden)
        .listRowInsets(Self.rowInsets)
    }

    @ViewBuilder
    private var history: some View {
        let sections = ChatStore.grouped(query.isEmpty ? store.conversations : store.search(query))
            .map { HistorySection(title: $0.title, items: $0.items) }
        if sections.isEmpty && !query.isEmpty {
            Text("No chats match \u{201C}\(query)\u{201D}")
                .font(.subheadline)
                .foregroundStyle(Theme.secondaryInk)
                .padding(.top, 16)
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)
                .listRowInsets(Self.rowInsets)
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
                .listRowInsets(Self.rowInsets)
            ForEach(section.items) { conversation in
                conversationRow(conversation)
            }
        }
    }

    private func conversationRow(_ conversation: Conversation) -> some View {
        let isOpen = conversation.id == chat.conversation.id
        return SidebarConversationRow(conversation: conversation, isOpen: isOpen) {
            chat.open(conversation.id)
            router.isSidebarOpen = false
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
        .listRowBackground(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(isOpen ? Theme.accentWash : Color.clear)
                .padding(.horizontal, 8)
        )
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
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Pie Voice settings")
    }

    // MARK: - Actions

    private func startChat(temporary: Bool) {
        chat.newChat(temporary: temporary)
        router.isSidebarOpen = false
    }

    private func rename(_ conversation: Conversation) {
        let title = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { return }
        store.rename(conversation.id, to: title)
    }

    private func delete(_ conversation: Conversation) {
        store.delete(conversation.id)
        // The controller would save the open thread again on its next
        // change; start a fresh one instead, as ChatGPT does.
        if conversation.id == chat.conversation.id {
            chat.newChat()
        }
    }

    private func isPresenting(_ item: Binding<Conversation?>) -> Binding<Bool> {
        Binding(get: { item.wrappedValue != nil }, set: { if !$0 { item.wrappedValue = nil } })
    }

    private static let rowInsets = EdgeInsets(top: 0, leading: 20, bottom: 0, trailing: 16)
}

private struct HistorySection: Identifiable {
    let title: String
    let items: [Conversation]

    var id: String { title }
}
