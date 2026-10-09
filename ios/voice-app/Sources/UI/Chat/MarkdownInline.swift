import SwiftUI

/// Inline Markdown (bold, italic, inline code, links, strikethrough) as an
/// `AttributedString` for `Text`, in the theme's colors.
enum MarkdownInline {
    /// - Parameter cached: keep the result for next time. Finished text is
    ///   drawn again whenever the list redraws, so it is cached; the end of
    ///   a streaming reply changes with every word and would only push
    ///   finished paragraphs out of the cache.
    static func render(_ source: String, cached: Bool = true) -> AttributedString {
        let key = source as NSString
        if cached, let hit = cache.object(forKey: key) { return hit.value }
        let options = AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: false,
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible
        )
        var text = (try? AttributedString(markdown: source, options: options)) ?? AttributedString(source)
        applyTheme(to: &text)
        if cached { cache.setObject(Rendered(text), forKey: key) }
        return text
    }

    /// While a reply streams, `**bold so far` shows its asterisks until the
    /// closing pair arrives, then snaps to bold. Closing the open span (or
    /// dropping a marker nothing has followed yet) renders the text the way
    /// it is going to end up. The same goes for `*italic so far` and a link
    /// whose address is still arriving (`[label](https://exa`).
    static func closingOpenSpans(_ source: String) -> String {
        var text = source
        for marker in ["**", "`"] {
            guard (text.components(separatedBy: marker).count - 1) % 2 == 1 else { continue }
            while let last = text.last, last.isWhitespace { text.removeLast() }
            if text.hasSuffix(marker) {
                text.removeLast(marker.count)
            } else {
                text += marker
            }
        }
        text = closingOpenItalic(text)
        if let open = text.range(of: "](", options: .backwards), !text[open.upperBound...].contains(")") {
            text += ")"
        }
        return text
    }

    /// A single `*` that opens emphasis (at the start of a word, with a
    /// word after it) and has no partner yet gets one at the end. A `*`
    /// between spaces or inside a word (`2 * 3`, `2*3`) is left alone:
    /// it may be arithmetic, and closing it would italicize the rest.
    private static func closingOpenItalic(_ text: String) -> String {
        let characters = Array(text)
        var singles: [Int] = []
        var index = 0
        while index < characters.count {
            if characters[index] == "*" {
                if index + 1 < characters.count, characters[index + 1] == "*" {
                    index += 2
                    continue
                }
                singles.append(index)
            }
            index += 1
        }
        guard singles.count % 2 == 1, let last = singles.last else { return text }
        let before = last > 0 ? characters[last - 1] : " "
        guard last + 1 < characters.count else {
            // A lone `*` with nothing after it yet.
            return String(characters[..<last])
        }
        let after = characters[last + 1]
        guard !after.isWhitespace, before.isWhitespace || before.isPunctuation else { return text }
        var closed = text
        while let end = closed.last, end.isWhitespace { closed.removeLast() }
        return closed + "*"
    }

    private static func applyTheme(to text: inout AttributedString) {
        for run in text.runs {
            if run.link != nil {
                text[run.range].swiftUI.foregroundColor = Theme.accent
                text[run.range].swiftUI.underlineStyle = .single
            }
            if let intent = run.inlinePresentationIntent, intent.contains(.code) {
                text[run.range].swiftUI.font = .system(.callout, design: .monospaced)
                text[run.range].swiftUI.backgroundColor = Theme.surfaceStrong
            }
        }
    }

    private final class Rendered {
        let value: AttributedString
        init(_ value: AttributedString) { self.value = value }
    }

    private static let cache: NSCache<NSString, Rendered> = {
        let cache = NSCache<NSString, Rendered>()
        cache.countLimit = 1024
        return cache
    }()
}
