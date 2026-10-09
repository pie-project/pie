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
/// themselves: a new bubble rises in, the pending dot pops in, removed
/// rows fade out on their own transition. This list does the same with
/// its own scroll-to-bottom button, which often comes or goes in the
/// middle of a glide. The list's glide is then the only animation moving
/// the content.
///
/// The keyboard is the other thing that changes the list's size during a
/// landing: a send dismisses it, and on a phone it slides down for a
/// quarter of a second while the question glides (an edit's sheet takes
/// its keyboard down the same way). So while a question lands, the room
/// below it is sized from the list's height with the keyboard down
/// (`RestingHeight`), not from its height at that moment: the keyboard
/// leaving then changes nothing in the transcript, the glide's end is
/// reachable from its first frame, and it lands once. (Recorded for an
/// edit: its room was laid out with the sheet's keyboard still up, the
/// list 349 pt tall, and the keyboard went 18 ms later; the transcript's
/// height did not change with it.)
///
/// A regenerate or an edit removes rows, often a long reply the user has
/// scrolled to the end of. The content would get shorter at once, the
/// scroll view would clamp to its new end, and the question would jump to
/// the top in one frame (recorded). So while a question lands, the
/// transcript does not get shorter (`TranscriptLayout.holdsHeight`): the
/// removed reply fades out where it was as the list glides up, and only
/// once the question is at the top does the empty space below it go.
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
    /// Set while a question lands (see above). A token, so the end of an
    /// earlier landing does not cut a later one short.
    @State private var landing: UUID?

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
    /// How long the transcript keeps its height for a landing: the glide
    /// (`Motion.scroll`) has settled to well under a point by then, and a
    /// phone's keyboard (a quarter of a second) is long down.
    private static let landingHold: TimeInterval = 0.8

    var body: some View {
        GeometryReader { outer in
            ScrollViewReader { proxy in
                transcript(
                    width: outer.size.width,
                    visibleHeight: outer.size.height,
                    restingHeight: RestingHeight.height(visible: outer.size)
                )
                    // A new scroll view per conversation: it starts at the
                    // end, and swaps in at once under the closing sidebar.
                    .id(chat.conversation.id)
                    .transition(.identity)
                    .modifier(EndTracker { old, new in
                        trackEnd(from: old, to: new)
                    })
                    .overlay(alignment: .bottom) {
                        if showsScrollButton {
                            ScrollToBottomButton {
                                withAnimation(reduceMotion ? nil : Motion.scroll) {
                                    proxy.scrollTo(Self.bottomID, anchor: .bottom)
                                }
                            }
                            .padding(.bottom, 8)
                            .transition(Self.scrollButtonTransition)
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
                    .onChange(of: outer.size, initial: true) { _, size in
                        RestingHeight.note(size)
                    }
            }
        }
        .environment(\.canRegenerate, chat.engineState == .ready && !chat.isGenerating)
        .sheet(item: $sheet) { sheet in
            switch sheet {
            case .edit(let message):
                EditMessageSheet(original: message.text) { newText in
                    ChatHaptics.messageSent(enabled: settings.haptics)
                    // Before the edit removes the turns after the message.
                    holdHeightForLanding()
                    // Stops a reply still on its way first, as ChatGPT does.
                    chat.edit(message.id, to: newText)
                }
            case .selectText(let text):
                SelectTextSheet(text: text)
            }
        }
    }

    /// - Parameters:
    ///   - visibleHeight: the height of the area the transcript is seen
    ///     in, below the top bar. (The scroll view itself reaches up under
    ///     the bar; this list's own frame does not.)
    ///   - restingHeight: that height with the keyboard down and the
    ///     composer at rest; the room a landing question gets (see above).
    private func transcript(width: CGFloat, visibleHeight: CGFloat, restingHeight: CGFloat) -> some View {
        let messages = chat.conversation.messages.filter { $0.role != .system }
        let pinnedID = liveQuestionID ?? pinnedQuestionID
        let isLanding = landing != nil && Self.canHoldHeight
        // Never less than what is visible, so the question can always
        // reach the top. Once the landing is over the room shrinks back to
        // the visible height, which leaves the question where it is: the
        // end the scroll view can reach is then exactly where it landed.
        let room = isLanding ? max(visibleHeight, restingHeight) : visibleHeight
        return ScrollView {
            VStack(spacing: 0) {
                TranscriptLayout(
                    spacing: Self.rowSpacing,
                    pinnedRow: pinnedID.flatMap { id in messages.firstIndex { $0.id == id } },
                    minPinnedHeight: room - Self.bottomPadding - Self.bottomMarkerHeight,
                    holdsHeight: isLanding
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

    /// The scroll-to-bottom button pops in and fades out on its own
    /// animations, so showing or hiding it, often in the middle of a
    /// glide, is not an animated change to the list (see above).
    private static var scrollButtonTransition: AnyTransition {
        .asymmetric(
            insertion: .opacity.combined(with: .scale(scale: Motion.prefersReduced ? 1 : 0.85))
                .animation(.snappy(duration: 0.22)),
            removal: .opacity.animation(Motion.fadeOut)
        )
    }

    private var actions: MessageActions {
        let chat = self.chat
        let holdHeightForLanding = self.holdHeightForLanding
        return MessageActions(
            setFeedback: { feedback, id in chat.setFeedback(feedback, for: id) },
            toggleReadAloud: { id in chat.toggleReadAloud(id) },
            regenerate: { id, mode in
                // Before the old reply is removed (see above).
                holdHeightForLanding()
                chat.regenerate(id, mode: mode)
            }
        )
    }

    /// Keeps the transcript from getting shorter until the question that
    /// is about to be landed has glided into place. Called before a
    /// regenerate or an edit removes rows, so the very first layout after
    /// the removal already holds; and by every landing.
    private func holdHeightForLanding() {
        let token = UUID()
        landing = token
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.landingHold) {
            if landing == token { landing = nil }
        }
    }

    /// Only on iOS 18. iOS 17's bottom scroll anchor also applies when the
    /// content changes size, so the held height going at the end of the
    /// glide would move the question down there.
    private static var canHoldHeight: Bool {
        if #available(iOS 18.0, *) { return true }
        return false
    }

    /// The question whose reply is being generated, if any.
    private var liveQuestionID: UUID? {
        let messages = chat.conversation.messages
        guard let reply = messages.lastIndex(where: \.isStreaming) else { return nil }
        return messages[..<reply].last { $0.role == .user }?.id
    }

    /// A reply has started: glide its question to the top. On the next
    /// turn of the run loop, so the room below it is laid out first.
    ///
    /// The scroll-to-bottom button fades as the glide starts: the glide is
    /// the way to the newest message now. (Left up, it stayed through the
    /// whole landing of an edit or of a send made while scrolled up, the
    /// end being far below for as long as the room and the held height
    /// lasted; recorded.)
    private func land(_ question: UUID?, _ proxy: ScrollViewProxy) {
        guard let question else { return }
        holdHeightForLanding()
        pinnedQuestionID = question
        scrollProbe.landedAt = Date()
        // Not animated: the button's transition carries its fade.
        if showsScrollButton { showsScrollButton = false }
        let animation = reduceMotion ? nil : Motion.scroll
        DispatchQueue.main.async {
            withAnimation(animation) {
                proxy.scrollTo(question, anchor: .top)
            }
        }
    }

    /// Shows or hides the scroll-to-bottom button from how far the end of
    /// the conversation is below the visible area.
    private func trackEnd(from old: EndGeometry, to new: EndGeometry) {
        scrollProbe.record(from: old, to: new)
        let distance = new.distance
        let shows: Bool
        if distance > Self.showButtonDistance {
            shows = true
        } else if distance < Self.hideButtonDistance {
            shows = false
        } else {
            return
        }
        guard shows != showsScrollButton else { return }
        // Not while a question lands: the room below it, and the height a
        // regenerate holds, put the end far below for a moment.
        if shows, landing != nil { return }
        if shows, let landed = scrollProbe.landedAt, -landed.timeIntervalSinceNow < Self.landingTime { return }
        // Not animated: the button's transition carries its animations.
        showsScrollButton = shows
    }

    /// The keyboard came up or the composer grew: if the end was in view,
    /// it stays in view, moving up with the keyboard (this runs inside the
    /// keyboard's own animation). A growing visible area needs nothing.
    ///
    /// "In view" is the same measure as the button's, taken before the
    /// visible area shrank (see `ScrollProbe.distance(beforeShrinkingBy:)`).
    private func keepEndInView(old: CGFloat, new: CGFloat, _ proxy: ScrollViewProxy) {
        guard new < old, scrollProbe.distance(beforeShrinkingBy: old - new) < Self.hideButtonDistance else { return }
        proxy.scrollTo(Self.bottomID, anchor: .bottom)
    }
}

/// The list's height with the keyboard down and the composer at rest:
/// the tallest it has been at this width. The keyboard, a taller composer
/// (more lines, an attachment) and the engine's boot pill above it only
/// ever make the list shorter, so its tallest is its resting height,
/// whatever the keyboard is doing now; a landing sizes its room from it
/// (see `MessageList`). Its height, not its position on screen: the list
/// rises 12 pt into place as the first message arrives, and a position
/// taken then was 12 pt off (recorded). Shared by every conversation's
/// list: a new chat's first list appears in the middle of a send, with
/// the keyboard on its way down.
@MainActor
private enum RestingHeight {
    private static var tallest: CGFloat = 0
    private static var width: CGFloat = 0

    static func note(_ size: CGSize) {
        guard size.height > 0 else { return }
        if abs(size.width - width) > 0.5 {
            // Another window size: start over.
            width = size.width
            tallest = size.height
        } else {
            tallest = max(tallest, size.height)
        }
    }

    /// The resting height of a list of size `visible`; its own height
    /// until one is known for its width.
    static func height(visible: CGSize) -> CGFloat {
        guard abs(visible.width - width) <= 0.5 else { return visible.height }
        return max(visible.height, tallest)
    }
}

/// The transcript's rows top to bottom, like a `VStack`, except that the
/// pinned row and everything after it take up at least `minPinnedHeight`.
/// That is the room that lets a just-sent question scroll to the top of
/// the screen while its reply is still short; the reply grows down into
/// it instead of pushing anything.
///
/// While `holdsHeight` is set, it is also never shorter than it was last
/// placed: rows that are removed leave their space until the hold ends.
private struct TranscriptLayout: Layout {
    var spacing: CGFloat
    var pinnedRow: Int?
    var minPinnedHeight: CGFloat
    var holdsHeight = false

    struct Cache {
        /// The height the rows were last placed in.
        var placedHeight: CGFloat = 0
    }

    func makeCache(subviews: Subviews) -> Cache {
        Cache()
    }

    /// Rows came or went. The default would start a new cache, and with it
    /// forget the height a hold keeps; the cache holds nothing about the
    /// rows themselves, so it is kept as it is.
    func updateCache(_ cache: inout Cache, subviews: Subviews) {}

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache) -> CGSize {
        let width = proposal.width ?? subviews.map { $0.sizeThatFits(.unspecified).width }.max() ?? 0
        let heights = rowHeights(subviews, width: width)
        var height = heights.reduce(0, +) + spacing * CGFloat(max(0, subviews.count - 1))
        if let pinnedRow, pinnedRow < subviews.count {
            let pinnedTop = heights[..<pinnedRow].reduce(0, +) + spacing * CGFloat(pinnedRow)
            height = max(height, pinnedTop + minPinnedHeight)
        }
        if holdsHeight {
            height = max(height, cache.placedHeight)
        }
        return CGSize(width: width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache) {
        cache.placedHeight = bounds.height
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

/// How far the end of the conversation is below the visible area, and
/// that area's height.
///
/// On iOS 18 from the scroll view's own geometry: how far it can still
/// scroll down, `contentSize + contentInsets.bottom - visibleRect.maxY`.
/// `visibleRect` is in the content's coordinates, so the top bar's inset
/// (the scroll view reaches up under the bar) plays no part, whatever
/// `containerSize` reports. Before iOS 18 from the content's frame in the
/// scroll view. Measured on iOS 26: that preference reported once, as
/// zero, and never again, so it is only the fallback.
private struct EndTracker: ViewModifier {
    let changed: (_ old: EndGeometry, _ new: EndGeometry) -> Void

    /// Before iOS 18: the last value reported, to pass as the old one.
    @State private var last = EndGeometry(distance: 0, visibleHeight: 0)

    func body(content: Content) -> some View {
        if #available(iOS 18.0, *) {
            content.onScrollGeometryChange(for: EndGeometry.self) { geometry in
                EndGeometry(
                    distance: geometry.contentSize.height + geometry.contentInsets.bottom
                        - geometry.visibleRect.maxY,
                    visibleHeight: geometry.visibleRect.height
                )
            } action: { old, new in
                changed(old, new)
            }
        } else {
            GeometryReader { outer in
                content.onPreferenceChange(ContentFrameKey.self) { frame in
                    let new = EndGeometry(
                        distance: frame.maxY - outer.size.height,
                        visibleHeight: outer.size.height
                    )
                    changed(last, new)
                    last = new
                }
            }
        }
    }
}

private struct EndGeometry: Equatable {
    var distance: CGFloat
    var visibleHeight: CGFloat
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

/// How far the end was below the visible area at the last scroll update,
/// and when the last question was landed. A class, not state: it changes
/// on every frame of a scroll, and only the button appearing or going
/// should redraw the list.
private final class ScrollProbe {
    private(set) var end = EndGeometry(distance: 0, visibleHeight: 0)
    /// The end as it was just before the visible area last got shorter,
    /// and when that was.
    private var beforeShrink: (end: EndGeometry, at: Date)?
    var landedAt: Date?

    func record(from old: EndGeometry, to new: EndGeometry) {
        if new.visibleHeight < old.visibleHeight - 0.5 {
            beforeShrink = (old, Date())
        } else if new.visibleHeight != old.visibleHeight {
            beforeShrink = nil
        }
        end = new
    }

    /// The distance to the end before the visible area got `shrink`
    /// points shorter. The list hears of the new height either before the
    /// scroll view reports it or just after; either way this is the
    /// distance from before.
    func distance(beforeShrinkingBy shrink: CGFloat) -> CGFloat {
        if let (before, at) = beforeShrink, -at.timeIntervalSinceNow < 0.5,
           abs((before.visibleHeight - end.visibleHeight) - shrink) < 1 {
            beforeShrink = nil
            return before.distance
        }
        return end.distance
    }
}

/// The conversation's frame in the scroll view's own coordinates.
private struct ContentFrameKey: PreferenceKey {
    static let defaultValue = CGRect.zero

    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        value = nextValue()
    }
}
