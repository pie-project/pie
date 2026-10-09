import SwiftUI

/// One parsed Markdown block. Lists and quotes hold blocks of their own,
/// so this view nests itself.
///
/// Equatable on its values, so while a reply streams only the block that
/// grew is drawn again.
struct MarkdownBlockView: View, Equatable {
    let block: MarkdownBlock
    /// Part of a reply that is still streaming: its text fades new words
    /// in, and a block that appears fades in.
    var isLive = false
    /// The last block of a reply that is still being written: it closes
    /// the spans the model has left open, and appeared just now if it is
    /// new.
    var isStreamingTail = false
    var listDepth = 0

    var body: some View {
        switch block {
        case .heading(let level, let text):
            inlineText(text)
                .font(Self.headingFont(level))
                .padding(.top, level <= 2 ? 6 : 2)
                .accessibilityAddTraits(.isHeader)
        case .paragraph(let text):
            inlineText(text)
                .font(.body)
                .lineSpacing(4)
        case .list(let list):
            listView(list)
        case .quote(let blocks):
            nestedBlocks(blocks, spacing: 10)
                .foregroundStyle(Theme.secondaryInk)
                .padding(.leading, 15)
                .background(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 1.5).fill(Theme.surfaceStrong).frame(width: 3)
                }
        case .code(let language, let code):
            CodeBlockView(language: language, code: code, isLive: isLive, startsFresh: isStreamingTail)
        case .rule:
            Rectangle().fill(Theme.hairline).frame(height: 1).padding(.vertical, 6)
                .modifier(FadeInWhenNew(isNew: isLive && isStreamingTail))
        case .table(let table):
            MarkdownTableView(table: table)
                .modifier(FadeInWhenNew(isNew: isLive && isStreamingTail))
        }
    }

    // MARK: - Pieces

    private func inlineText(_ source: String) -> RevealText {
        // The end of a streaming reply changes with every word: closed
        // spans, and not cached. It is also the paragraph that wraps as it
        // grows, so it ends in invisible words that keep its words from
        // jumping between lines (see `RevealText`).
        let attributed = isStreamingTail
            ? MarkdownInline.render(MarkdownInline.closingOpenSpans(source), cached: false)
            : MarkdownInline.render(source)
        return RevealText(attributed, isLive: isLive, startsFresh: isStreamingTail, padsLastLine: isStreamingTail)
    }

    private func listView(_ list: MarkdownList) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(list.items.enumerated()), id: \.offset) { offset, item in
                let isLastItem = offset == list.items.count - 1
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    // A new item's marker fades in on the same clock and
                    // curve as its first words, by going through the same
                    // renderer; with a fade of its own it trailed them by
                    // a few frames (recorded).
                    RevealText(
                        AttributedString(marker(for: list, at: offset)),
                        isLive: isLive,
                        startsFresh: isStreamingTail && isLastItem
                    )
                    .font(.body)
                    .monospacedDigit()
                    .frame(minWidth: list.isOrdered ? 20 : 12, alignment: .trailing)
                    .accessibilityHidden(!list.isOrdered)
                    VStack(alignment: .leading, spacing: 8) {
                        MarkdownBlockView(
                            block: .paragraph(item.text),
                            isLive: isLive,
                            isStreamingTail: isStreamingTail && isLastItem && item.children.isEmpty,
                            listDepth: listDepth
                        )
                        ForEach(Array(item.children.enumerated()), id: \.offset) { childOffset, child in
                            MarkdownBlockView(
                                block: child,
                                isLive: isLive,
                                isStreamingTail: isStreamingTail && isLastItem
                                    && childOffset == item.children.count - 1,
                                listDepth: listDepth + 1
                            )
                        }
                    }
                }
            }
        }
    }

    private func nestedBlocks(_ blocks: [MarkdownBlock], spacing: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: spacing) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { offset, child in
                MarkdownBlockView(
                    block: child,
                    isLive: isLive,
                    isStreamingTail: isStreamingTail && offset == blocks.count - 1,
                    listDepth: listDepth
                )
            }
        }
    }

    private func marker(for list: MarkdownList, at offset: Int) -> String {
        if list.isOrdered { return "\(list.start + offset)." }
        switch listDepth {
        case 0: return "\u{2022}"
        case 1: return "\u{25E6}"
        default: return "\u{25AA}"
        }
    }

    /// The site sets its headings in bold serif; a reply's top-level
    /// headings follow it, smaller ones stay in the body face.
    private static func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .system(.title2, design: .serif, weight: .bold)
        case 2: return .system(.title3, design: .serif, weight: .bold)
        case 3: return .headline
        default: return .system(.subheadline, weight: .semibold)
        }
    }
}

/// A block that appears while a reply streams (a code block's frame, a
/// table, a rule) fades in instead of popping in,
/// as the words in it do. One that is already there when the view is
/// built (a finished reply, an older block) just shows.
struct FadeInWhenNew: ViewModifier {
    @State private var isShown: Bool

    init(isNew: Bool) {
        _isShown = State(initialValue: !isNew)
    }

    func body(content: Content) -> some View {
        content
            .opacity(isShown ? 1 : 0)
            .onAppear {
                guard !isShown else { return }
                withAnimation(Motion.fadeIn) { isShown = true }
            }
    }
}
