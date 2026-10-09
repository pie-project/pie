import SwiftUI

/// Inline Markdown (bold, italic, inline code, links, strikethrough) as an
/// `AttributedString` for `Text`, in the theme's colors.
enum MarkdownInline {
    static func render(_ source: String) -> AttributedString {
        let key = source as NSString
        if let cached = cache.object(forKey: key) { return cached.value }
        let options = AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: false,
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible
        )
        var text = (try? AttributedString(markdown: source, options: options)) ?? AttributedString(source)
        applyTheme(to: &text)
        cache.setObject(Rendered(text), forKey: key)
        return text
    }

    /// ChatGPT's streaming dot, set inline after the last word so it
    /// follows the text as it wraps.
    static var streamingDot: AttributedString {
        var dot = AttributedString("\u{2009}\u{25CF}")
        dot.swiftUI.font = .system(size: 11)
        dot.swiftUI.baselineOffset = 1
        dot.swiftUI.foregroundColor = Theme.ink
        return dot
    }

    /// While a reply streams, `**bold so far` shows its asterisks until the
    /// closing pair arrives, then snaps to bold. Closing the open span (or
    /// dropping a marker nothing has followed yet) renders the text the way
    /// it is going to end up.
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
        return text
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
