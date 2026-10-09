import SwiftUI

/// The open conversation: user bubbles and Markdown replies.
///
/// Scrolling works the way ChatGPT's does since 2025:
///
/// - Sending (or editing, or regenerating) glides the question to the top
///   of the screen, wherever the list was, and the reply grows down into
///   the space below it. The list leaves room for that: the newest
///   question and everything after it are at least a screen tall
///   (`TranscriptLayout`).
/// - The list never follows the reply as it grows, so the start of the
///   answer stays put while it is read. Once the end of the conversation
///   is out of sight below, a button offers a glide back down to it.
/// - A conversation opens at its end, at once, with no scroll animation:
///   each conversation gets a fresh scroll view whose starting position is
///   the bottom.
///
/// Measured in the iOS 26 Simulator: a `scrollTo` made while the scroll
/// view's content is laying out inside an animation (a send wrapped in
/// `withAnimation`) is dropped, and the question never lands. So the
/// controller changes the rows without an animation and the rows animate
/// themselves: a new bubble rises in, the pending dot pops in, removed rows
/// fade out on their own transition. The list's glide is then the only
/// animation moving the content.
///
/// The stack is not lazy. A lazy stack scrolled programmatically to a row
/// it has not built yet can land on an empty screen. Rows compare equal
/// unless their message changed, so a streamed word redraws only the reply
/// it belongs to.
struct MessageList: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The question last landed at the top. Kept after its reply has
    /// finished, so the room below it stays and nothing jumps.
    @State private var pinnedQuestionID: UUID?
    @State private var showsScrollButton = false
    @State private var scrollProbe = ScrollProbe()
    @State private var sheet: MessageSheet?

    private static let bottomID = "message-list-bottom"
    private static let space = "message-list"
    private static let rowSpacing: CGFloat = 24
    private static let bottomPadding: CGFloat = 12
    private static let bottomMarkerHeight: CGFloat = 1
    /// The scroll-to-bottom button appears once the end is this far below
    /// the visible area, and goes again once it is this close, so it does
    /// not flicker around one threshold.
    private static let showButtonDistance: CGFloat = 80
    private static let hideButtonDistance: CGFloat = 20
    /// How long a landing glide takes to settle. The room left below a
    /// new question puts the end far below for a moment, which must not
    /// flash the button.
    private static let landingTime: TimeInterval = 0.6

    var body: some View {
        GeometryReader { outer in
            ScrollViewReader { proxy in
                transcript(width: outer.size.width, visibleHeight: outer.size.height)
                    // A new scroll view per conversation: it starts at the
                    // end, and swaps in at once under the closing sidebar.
                    .id(chat.conversation.id)
                    .transition(.identity)
                    .modifier(EndTracker { contentBottom, height in
                        trackEnd(contentBottom: contentBottom, height: height)
                    })
                    .overlay(alignment: .bottom) {
                        if showsScrollButton {
                            ScrollToBottomButton {
                                withAnimation(reduceMotion ? nil : Motion.scroll) {
                                    proxy.scrollTo(Self.bottomID, anchor: .bottom)
                                }
                            }
                            .padding(.bottom, 8)
                            .transition(.popIn(from: 0.85))
                        }
                    }
                    .onChange(of: liveQuestionID, initial: true) { _, question in
                        land(question, proxy)
                    }
                    .onChange(of: chat.conversation.id) { _, _ in
                        pinnedQuestionID = nil
                        showsScrollButton = false
                        scrollProbe = ScrollProbe()
                    }
                    .onChange(of: outer.size.height) { old, new in
                        keepEndInView(old: old, new: new, proxy)
                    }
            }
        }
        .environment(\.canRegenerate, chat.engineState == .ready && !chat.isGenerating)
        .sheet(item: $sheet) { sheet in
            switch sheet {
            case .edit(let message):
                EditMessageSheet(original: message.text) { newText in
                    ChatHaptics.messageSent(enabled: settings.haptics)
                    // Stops a reply still on its way first, as ChatGPT does.
                    chat.edit(message.id, to: newText)
                }
            case .selectText(let text):
                SelectTextSheet(text: text)
            }
        }
    }

    /// - Parameter visibleHeight: the height of the area the transcript is
    ///   seen in, below the top bar. (The scroll view itself reaches up
    ///   under the bar; this list's own frame does not.)
    private func transcript(width: CGFloat, visibleHeight: CGFloat) -> some View {
        let messages = chat.conversation.messages.filter { $0.role != .system }
        let pinnedID = liveQuestionID ?? pinnedQuestionID
        return ScrollView {
            VStack(spacing: 0) {
                TranscriptLayout(
                    spacing: Self.rowSpacing,
                    pinnedRow: pinnedID.flatMap { id in messages.firstIndex { $0.id == id } },
                    minPinnedHeight: visibleHeight - Self.bottomPadding - Self.bottomMarkerHeight
                ) {
                    ForEach(messages) { message in
                        // An explicit id: `scrollTo` did not find the rows of
                        // a custom layout by their `ForEach` identity alone.
                        row(for: message, width: width)
                            .id(message.id)
                    }
                }
                Color.clear
                    .frame(height: Self.bottomMarkerHeight)
                    .id(Self.bottomID)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, Self.bottomPadding)
            .background {
                // Before iOS 18, where the end is tracked through this.
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
        .modifier(StartsAtEnd())
    }

    @ViewBuilder
    private func row(for message: StoredMessage, width: CGFloat) -> some View {
        switch message.role {
        case .user:
            // The row rises into place itself (see `UserMessageRow`).
            UserMessageRow(
                message: message,
                minimumLeadingSpace: (width - 32) * 0.22,
                canEdit: chat.engineState == .ready,
                sheet: $sheet
            )
            .equatable()
            .transition(.asymmetric(insertion: .identity, removal: Self.leaving))
        case .assistant:
            AssistantMessageRow(
                message: message,
                livePhase: message.isStreaming ? chat.phase : nil,
                isReadingAloud: chat.readingAloudMessageID == message.id,
                showsStats: settings.showEngineStats,
                actions: actions,
                sheet: $sheet
            )
            .equatable()
            // Its pending dot pops in by itself, a beat after the bubble.
            .transition(.asymmetric(insertion: .identity, removal: Self.leaving))
        case .system:
            EmptyView()
        }
    }

    /// A row removed by a regenerate or an edit fades out where it is,
    /// with its own animation (the controller makes those changes without
    /// one; see above). It no longer takes up space, so what replaces it
    /// fades in in the same place.
    private static let leaving = AnyTransition.opacity.animation(Motion.fadeOut)

    private var actions: MessageActions {
        let chat = self.chat
        return MessageActions(
            setFeedback: { feedback, id in chat.setFeedback(feedback, for: id) },
            toggleReadAloud: { id in chat.toggleReadAloud(id) },
            regenerate: { id, mode in chat.regenerate(id, mode: mode) }
        )
    }

    /// The question whose reply is being generated, if any.
    private var liveQuestionID: UUID? {
        let messages = chat.conversation.messages
        guard let reply = messages.lastIndex(where: \.isStreaming) else { return nil }
        return messages[..<reply].last { $0.role == .user }?.id
    }

    /// A reply has started: glide its question to the top. On the next
    /// turn of the run loop, so the room below it is laid out first.
    private func land(_ question: UUID?, _ proxy: ScrollViewProxy) {
        guard let question else { return }
        pinnedQuestionID = question
        scrollProbe.landedAt = Date()
        let animation = reduceMotion ? nil : Motion.scroll
        DispatchQueue.main.async {
            withAnimation(animation) {
                proxy.scrollTo(question, anchor: .top)
            }
        }
    }

    /// Shows or hides the scroll-to-bottom button from how far the end of
    /// the conversation is below the visible area.
    ///
    /// - Parameters:
    ///   - contentBottom: where the end of the content is, measured from the
    ///     top of the visible area.
    ///   - height: the visible area's height.
    private func trackEnd(contentBottom: CGFloat, height: CGFloat) {
        scrollProbe.contentBottom = contentBottom
        let distance = contentBottom - height
        let shows: Bool
        if distance > Self.showButtonDistance {
            shows = true
        } else if distance < Self.hideButtonDistance {
            shows = false
        } else {
            return
        }
        guard shows != showsScrollButton else { return }
        if shows, let landed = scrollProbe.landedAt, -landed.timeIntervalSinceNow < Self.landingTime { return }
        withAnimation(shows ? .snappy(duration: 0.22) : Motion.fadeOut) {
            showsScrollButton = shows
        }
    }

    /// The keyboard came up or the composer grew: if the end was in view,
    /// it stays in view, moving up with the keyboard (this runs inside the
    /// keyboard's own animation). A growing visible area needs nothing.
    private func keepEndInView(old: CGFloat, new: CGFloat, _ proxy: ScrollViewProxy) {
        guard new < old, scrollProbe.contentBottom - old < Self.hideButtonDistance else { return }
        proxy.scrollTo(Self.bottomID, anchor: .bottom)
    }
}

/// The transcript's rows top to bottom, like a `VStack`, except that the
/// pinned row and everything after it take up at least `minPinnedHeight`.
/// That is the room that lets a just-sent question scroll to the top of
/// the screen while its reply is still short; the reply grows down into
/// it instead of pushing anything.
private struct TranscriptLayout: Layout {
    var spacing: CGFloat
    var pinnedRow: Int?
    var minPinnedHeight: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? subviews.map { $0.sizeThatFits(.unspecified).width }.max() ?? 0
        let heights = rowHeights(subviews, width: width)
        var height = heights.reduce(0, +) + spacing * CGFloat(max(0, subviews.count - 1))
        if let pinnedRow, pinnedRow < subviews.count {
            let pinnedTop = heights[..<pinnedRow].reduce(0, +) + spacing * CGFloat(pinnedRow)
            height = max(height, pinnedTop + minPinnedHeight)
        }
        return CGSize(width: width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let heights = rowHeights(subviews, width: bounds.width)
        var y = bounds.minY
        for (subview, height) in zip(subviews, heights) {
            subview.place(
                at: CGPoint(x: bounds.minX, y: y),
                anchor: .topLeading,
                proposal: ProposedViewSize(width: bounds.width, height: height)
            )
            y += height + spacing
        }
    }

    private func rowHeights(_ subviews: Subviews, width: CGFloat) -> [CGFloat] {
        subviews.map { $0.sizeThatFits(ProposedViewSize(width: width, height: nil)).height }
    }
}

/// Reports where the end of the content is, measured from the top of the
/// visible area, and that area's height. On iOS 18 from the scroll view's
/// own geometry; before that from the content's frame in the scroll view.
/// Measured on iOS 26: the content-frame preference reported once, as
/// zero, and never again, so it is only the fallback.
///
/// The scroll view reaches up under the top bar and insets its content by
/// the bar's height (`contentInsets.top`); `containerSize` is the area
/// below that inset, and an offset of `-contentInsets.top` is the top.
private struct EndTracker: ViewModifier {
    let changed: (_ contentBottom: CGFloat, _ height: CGFloat) -> Void

    func body(content: Content) -> some View {
        if #available(iOS 18.0, *) {
            content.onScrollGeometryChange(for: EndGeometry.self) { geometry in
                EndGeometry(
                    contentBottom: geometry.contentSize.height
                        - geometry.contentOffset.y - geometry.contentInsets.top,
                    height: geometry.containerSize.height
                )
            } action: { _, end in
                changed(end.contentBottom, end.height)
            }
        } else {
            GeometryReader { outer in
                content.onPreferenceChange(ContentFrameKey.self) { frame in
                    changed(frame.maxY, outer.size.height)
                }
            }
        }
    }
}

private struct EndGeometry: Equatable {
    var contentBottom: CGFloat
    var height: CGFloat
}

/// Opens a conversation at its end. On iOS 18 only the starting position
/// is anchored; iOS 17 has just the one anchor, which also keeps the end
/// in view as the content changes.
private struct StartsAtEnd: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 18.0, *) {
            content.defaultScrollAnchor(.bottom, for: .initialOffset)
        } else {
            content.defaultScrollAnchor(.bottom)
        }
    }
}

/// Where the content's bottom edge was at the last scroll update, and
/// when the last question was landed. A class, not state: it changes on
/// every frame of a scroll, and only the button appearing or going should
/// redraw the list.
private final class ScrollProbe {
    var contentBottom: CGFloat = 0
    var landedAt: Date?
}

/// The conversation's frame in the scroll view's own coordinates.
private struct ContentFrameKey: PreferenceKey {
    static let defaultValue = CGRect.zero

    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        value = nextValue()
    }
}
