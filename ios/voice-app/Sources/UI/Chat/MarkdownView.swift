import SwiftUI

/// A reply's Markdown drawn with native views.
///
/// Equatable on its input: the list redraws every visible reply whenever
/// one of them streams, and a finished reply compares equal and is skipped.
struct MarkdownView: View, Equatable {
    let text: String
    /// Draws the streaming dot after the last character while the reply is
    /// still being written.
    var isStreaming = false

    var body: some View {
        let blocks = MarkdownParser.parse(text)
        VStack(alignment: .leading, spacing: 14) {
            // Blocks are keyed by position: while a reply streams only the
            // last one changes, so the ones above keep their identity and
            // are not redrawn.
            ForEach(Array(blocks.enumerated()), id: \.offset) { index, block in
                MarkdownBlockView(block: block, isStreamingTail: isStreaming && index == blocks.count - 1)
            }
            if isStreaming && blocks.isEmpty {
                StreamingDot()
            }
        }
        .foregroundStyle(Theme.ink)
        .tint(Theme.accent)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The streaming dot on a line of its own, after a block that has no text
/// to set it in (a code block, a table, a rule).
struct StreamingDot: View {
    var body: some View {
        Circle()
            .fill(Theme.ink)
            .frame(width: 10, height: 10)
            .accessibilityHidden(true)
    }
}
