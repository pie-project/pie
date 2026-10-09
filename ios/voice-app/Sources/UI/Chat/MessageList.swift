import SwiftUI

/// The open conversation: user bubbles and Markdown replies. It follows
/// the newest text while the user stays at the bottom; once they scroll up
/// it stays put and offers a button back down.
///
/// The stack is not lazy. A lazy stack scrolled programmatically to its
/// end (on opening a chat, and on every streamed token) can land on an
/// offset whose rows it never realises and show an empty screen. Rows
/// compare equal unless their message changed, so a token redraws only the
/// reply it belongs to.
struct MessageList: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var settings: AppSettings

    /// Whether new text scrolls the list. Content growing past the bottom
    /// never turns this off (a code block can arrive in one update); only
    /// the user scrolling up does, and reaching the end again turns it back
    /// on.
    @State private var isFollowing = true
    @State private var scrollProbe = ScrollProbe()
    @State private var sheet: MessageSheet?

    private static let bottomID = "message-list-bottom"
    private static let space = "message-list"
    /// How far below the visible area the end may be and still count as
    /// on screen.
    private static let endSlack: CGFloat = 24

    var body: some View {
        GeometryReader { outer in
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        ForEach(chat.conversation.messages) { message in
                            row(for: message, width: outer.size.width)
                        }
                        if showsPendingReply {
                            PendingReplyRow(phase: chat.phase)
                        }
                        Color.clear
                            .frame(height: 1)
                            .id(Self.bottomID)
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 16)
                    .padding(.bottom, 12)
                    .background {
                        GeometryReader { content in
                            Color.clear.preference(
                                key: ContentFrameKey.self,
                                value: content.frame(in: .named(Self.space))
                            )
                        }
                    }
                }
                .coordinateSpace(.named(Self.space))
                .scrollDismissesKeyboard(.interactively)
                .onPreferenceChange(ContentFrameKey.self) { frame in
                    trackScrolling(frame, viewportHeight: outer.size.height)
                }
                .overlay(alignment: .bottom) {
                    ZStack {
                        if !isFollowing {
                            ScrollToBottomButton {
                                isFollowing = true
                                withAnimation(.easeOut(duration: 0.25)) {
                                    proxy.scrollTo(Self.bottomID, anchor: .bottom)
                                }
                            }
                            .transition(.scale(scale: 0.6).combined(with: .opacity))
                        }
                    }
                    .padding(.bottom, 6)
                    .animation(.easeOut(duration: 0.18), value: isFollowing)
                }
                .onAppear {
                    scrollToBottom(proxy)
                }
                .onChange(of: chat.conversation.id) { _, _ in
                    isFollowing = true
                    scrollToBottom(proxy)
                }
                .onChange(of: tail) { old, new in
                    if new.count > old.count, new.lastIsUser {
                        // The user just sent: always bring their message
                        // into view, wherever they had scrolled to.
                        isFollowing = true
                        withAnimation(.easeOut(duration: 0.25)) {
                            proxy.scrollTo(Self.bottomID, anchor: .bottom)
                        }
                    } else if isFollowing {
                        proxy.scrollTo(Self.bottomID, anchor: .bottom)
                    }
                }
                .onChange(of: outer.size.height) { _, _ in
                    // The keyboard came up or went down.
                    if isFollowing { proxy.scrollTo(Self.bottomID, anchor: .bottom) }
                }
            }
        }
        .sheet(item: $sheet) { sheet in
            switch sheet {
            case .edit(let message):
                EditMessageSheet(original: message.text) { newText in
                    ChatHaptics.messageSent(enabled: settings.haptics)
                    // As in ChatGPT, sending an edit answers it at once: a
                    // reply still on its way is stopped first (the edit
                    // drops it anyway), since the controller takes an edit
                    // only while nothing is generating.
                    chat.stop()
                    chat.edit(message.id, to: newText)
                }
            case .selectText(let text):
                SelectTextSheet(text: text)
            }
        }
    }

    @ViewBuilder
    private func row(for message: StoredMessage, width: CGFloat) -> some View {
        switch message.role {
        case .user:
            UserMessageRow(
                message: message,
                minimumLeadingSpace: (width - 32) * 0.22,
                canEdit: chat.engineState == .ready,
                sheet: $sheet
            )
            .equatable()
        case .assistant:
            AssistantMessageRow(
                message: message,
                livePhase: message.isStreaming ? chat.phase : nil,
                isReadingAloud: chat.readingAloudMessageID == message.id,
                canRegenerate: chat.engineState == .ready && !chat.isGenerating,
                showsStats: settings.showEngineStats,
                actions: actions,
                sheet: $sheet
            )
            .equatable()
        case .system:
            EmptyView()
        }
    }

    private var actions: MessageActions {
        let chat = self.chat
        return MessageActions(
            setFeedback: { feedback, id in chat.setFeedback(feedback, for: id) },
            toggleReadAloud: { id in chat.toggleReadAloud(id) },
            regenerate: { id, mode in chat.regenerate(id, mode: mode) }
        )
    }

    /// A reply is under way but the controller has not added its message
    /// to the transcript yet.
    private var showsPendingReply: Bool {
        guard chat.isGenerating else { return false }
        guard let last = chat.conversation.messages.last else { return true }
        return !(last.role == .assistant && last.isStreaming)
    }

    /// The parts of the conversation's end that change its height.
    private var tail: Tail {
        let last = chat.conversation.messages.last
        return Tail(
            count: chat.conversation.messages.count,
            lastIsUser: last?.role == .user,
            textLength: last?.text.count ?? 0,
            reasoningLength: last?.reasoning.count ?? 0,
            isStreaming: last?.isStreaming ?? false,
            phase: chat.phase
        )
    }

    /// The content moving down in the viewport means the user scrolled up
    /// (growth only moves its bottom edge; scrolling to the end moves it up).
    private func trackScrolling(_ frame: CGRect, viewportHeight: CGFloat) {
        let movedDown = scrollProbe.lastContentTop.map { frame.minY > $0 + 1 } ?? false
        scrollProbe.lastContentTop = frame.minY
        if frame.maxY <= viewportHeight + Self.endSlack {
            if !isFollowing { isFollowing = true }
        } else if movedDown, isFollowing {
            isFollowing = false
        }
    }

    /// On a newly shown conversation the first scroll can run before the
    /// rows have their final heights; a second one on the next turn of the
    /// run loop lands on the real end.
    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        proxy.scrollTo(Self.bottomID, anchor: .bottom)
        DispatchQueue.main.async {
            proxy.scrollTo(Self.bottomID, anchor: .bottom)
        }
    }
}

private struct Tail: Equatable {
    var count: Int
    var lastIsUser: Bool
    var textLength: Int
    var reasoningLength: Int
    var isStreaming: Bool
    var phase: ChatController.ReplyPhase
}

/// Where the content's top edge was at the last scroll update. A class, not
/// state: it changes on every frame of a scroll, and only `isFollowing`
/// flipping should redraw the list.
private final class ScrollProbe {
    var lastContentTop: CGFloat?
}

/// The conversation's frame in the scroll view's own coordinates.
private struct ContentFrameKey: PreferenceKey {
    static let defaultValue = CGRect.zero

    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        value = nextValue()
    }
}
