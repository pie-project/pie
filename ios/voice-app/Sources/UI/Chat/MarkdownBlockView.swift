import SwiftUI

/// One parsed Markdown block. Lists and quotes hold blocks of their own,
/// so this view nests itself.
struct MarkdownBlockView: View {
    let block: MarkdownBlock
    /// The last block of a reply that is still being written: it carries
    /// the streaming dot and closes the spans the model has left open.
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
            withTrailingDot(CodeBlockView(language: language, code: code))
        case .rule:
            withTrailingDot(Rectangle().fill(Theme.hairline).frame(height: 1).padding(.vertical, 6))
        case .table(let table):
            withTrailingDot(MarkdownTableView(table: table))
        }
    }

    // MARK: - Pieces

    private func inlineText(_ source: String) -> Text {
        guard isStreamingTail else { return Text(MarkdownInline.render(source)) }
        var attributed = MarkdownInline.render(MarkdownInline.closingOpenSpans(source))
        attributed.append(MarkdownInline.streamingDot)
        return Text(attributed)
    }

    private func listView(_ list: MarkdownList) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(list.items.enumerated()), id: \.offset) { offset, item in
                let isLastItem = offset == list.items.count - 1
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(marker(for: list, at: offset))
                        .font(.body)
                        .monospacedDigit()
                        .frame(minWidth: list.isOrdered ? 20 : 12, alignment: .trailing)
                        .accessibilityHidden(!list.isOrdered)
                    VStack(alignment: .leading, spacing: 8) {
                        MarkdownBlockView(
                            block: .paragraph(item.text),
                            isStreamingTail: isStreamingTail && isLastItem && item.children.isEmpty,
                            listDepth: listDepth
                        )
                        ForEach(Array(item.children.enumerated()), id: \.offset) { childOffset, child in
                            MarkdownBlockView(
                                block: child,
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
                    isStreamingTail: isStreamingTail && offset == blocks.count - 1,
                    listDepth: listDepth
                )
            }
        }
    }

    @ViewBuilder
    private func withTrailingDot<Content: View>(_ content: Content) -> some View {
        if isStreamingTail {
            VStack(alignment: .leading, spacing: 10) {
                content
                StreamingDot()
            }
        } else {
            content
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
