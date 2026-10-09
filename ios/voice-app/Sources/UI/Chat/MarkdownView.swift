import SwiftUI

/// A reply's Markdown drawn with native views.
///
/// Equatable on its input: the list redraws every visible reply whenever
/// one of them streams, and a finished reply compares equal and is skipped.
///
/// While the reply streams, it is styled from the first word (bold is bold
/// before its closing `**` arrives), its last line is held back while it
/// could still turn into something else (`MarkdownParser.withoutUndecidedEnd`),
/// only the end of it is parsed again per reveal, and new words fade in
/// (`RevealText`). There is no trailing dot: as in ChatGPT, the pending dot
/// before the first words is the only one.
struct MarkdownView: View, Equatable {
    let text: String
    var isStreaming = false

    var body: some View {
        let blocks = isStreaming
            ? MarkdownParser.parseStreaming(MarkdownParser.withoutUndecidedEnd(text))
            : MarkdownParser.parse(text)
        VStack(alignment: .leading, spacing: 14) {
            // Blocks are keyed by position: while a reply streams only the
            // last one changes, so the ones above keep their identity and,
            // comparing equal, are not redrawn.
            ForEach(Array(blocks.enumerated()), id: \.offset) { index, block in
                MarkdownBlockView(
                    block: block,
                    isLive: isStreaming,
                    isStreamingTail: isStreaming && index == blocks.count - 1
                )
                .equatable()
            }
        }
        .foregroundStyle(Theme.ink)
        .tint(Theme.accent)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
