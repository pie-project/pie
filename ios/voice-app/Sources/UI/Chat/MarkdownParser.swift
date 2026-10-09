import Foundation

/// One block of a reply's Markdown.
indirect enum MarkdownBlock: Equatable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case list(MarkdownList)
    case quote([MarkdownBlock])
    /// A fenced code block. While a reply streams, a fence that has not
    /// been closed yet holds everything after it.
    case code(language: String, code: String)
    case rule
    case table(MarkdownTable)
}

struct MarkdownList: Equatable {
    var isOrdered: Bool
    /// The first item's number, for ordered lists.
    var start: Int
    var items: [MarkdownListItem]
}

struct MarkdownListItem: Equatable {
    var text: String
    /// Nested lists, code and paragraphs indented under the item.
    var children: [MarkdownBlock]
}

struct MarkdownTable: Equatable {
    enum Alignment: Equatable {
        case leading
        case center
        case trailing
    }

    var header: [String]
    var alignments: [Alignment]
    /// Each row padded or cut to the header's width.
    var rows: [[String]]
}

/// Parses the part of CommonMark and GitHub tables that a chat model
/// writes.
///
/// It reads line by line and never needs a line it has not been given,
/// so as a reply grows its earlier blocks stay as they were and only the
/// last one changes. That is what keeps the rendering from flickering
/// while the model writes, and what lets `parseStreaming` keep the
/// finished blocks and parse only the last one again. It is lenient
/// where models are sloppy: code inside a list item may be unindented,
/// and a table may be drawn as soon as its delimiter row starts.
enum MarkdownParser {
    /// A finished reply, parsed once and cached.
    static func parse(_ source: String) -> [MarkdownBlock] {
        let key = source as NSString
        if let cached = cache.object(forKey: key) { return cached.blocks }
        var parser = BlockParser(lines: lines(of: source).map(\.text))
        let blocks = parser.parse()
        cache.setObject(Parsed(blocks), forKey: key)
        return blocks
    }

    /// A reply that is still streaming, parsed again for every reveal.
    ///
    /// Text appended later can change only the last two blocks: the last
    /// one grows, and a last line still being written can join the block
    /// before it (`3` becoming `3.`, the next item of a list above a blank
    /// line). Every block before those is closed. So the closed blocks of
    /// the previous call are kept and only the text from the start of the
    /// second-to-last block on is parsed again, which keeps the work per
    /// reveal from growing with the reply. Nothing here goes into the
    /// cache of finished replies.
    @MainActor
    static func parseStreaming(_ source: String) -> [MarkdownBlock] {
        var closed: [MarkdownBlock] = []
        var resumeAt = 0
        if let memo = streamMemo,
           source.utf8.count >= memo.openStart,
           source.utf8.starts(with: memo.source.utf8.prefix(memo.openStart)) {
            closed = memo.closedBlocks
            resumeAt = memo.openStart
        }
        let rest = String(Substring(source.utf8.dropFirst(resumeAt)))
        let lines = lines(of: rest)
        var parser = BlockParser(lines: lines.map(\.text))
        let blocks = parser.parse()

        // The last two blocks are the part the next call parses again.
        // With no blocks yet, nothing is closed: start from the same place.
        let open = min(2, blocks.count)
        let openStart = open > 0 ? resumeAt + lines[parser.startLines[blocks.count - open]].start : resumeAt
        streamMemo = StreamMemo(
            source: source,
            closedBlocks: closed + blocks.dropLast(open),
            openStart: openStart
        )
        return closed + blocks
    }

    /// While a reply streams, its last line can be the start of something
    /// that is not decided yet: a lone `-` or `*` (a list item, a rule,
    /// bold text?), `#` (a heading), backticks (a code fence), or a line
    /// starting with `|` (a table's header, drawn as plain text with its
    /// pipes until the delimiter row under it starts). Showing those and
    /// then redrawing them as something else flickers; this leaves them
    /// out until the next words decide them. Finished replies are shown
    /// whole.
    static func withoutUndecidedEnd(_ source: String) -> String {
        guard let lastBreak = source.lastIndex(of: "\n") else {
            return isUndecided(source[...], previous: nil, beforePrevious: nil) ? "" : source
        }
        let last = source[source.index(after: lastBreak)...]
        let head = source[..<lastBreak]
        let previousBreak = head.lastIndex(of: "\n")
        let previous = previousBreak.map { head[head.index(after: $0)...] } ?? head
        let beforePrevious: Substring? = previousBreak.map { break_ in
            let earlier = head[..<break_]
            return earlier.lastIndex(of: "\n").map { earlier[earlier.index(after: $0)...] } ?? earlier
        }
        guard isUndecided(last, previous: previous, beforePrevious: beforePrevious) else { return source }
        // A delimiter row with no dash yet hides the header above it too.
        if isTableHeader(previous, before: beforePrevious), Line.isBareDelimiterStart(last) {
            return String(source[..<(previousBreak ?? source.startIndex)])
        }
        return String(head)
    }

    private static func isUndecided(_ line: Substring, previous: Substring?, beforePrevious: Substring?) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return false }
        if trimmed.allSatisfy({ "-*+_#`~=".contains($0) }) { return true }
        guard trimmed.hasPrefix("|") else { return false }
        if let previous, isTableHeader(previous, before: beforePrevious) {
            // The delimiter row under a header: undecided only until its
            // first dash arrives, when the table is drawn.
            return Line.isBareDelimiterStart(line)
        }
        if let previous, previous.trimmingCharacters(in: .whitespaces).hasPrefix("|") {
            // A row of a table already drawn; only a bare `|` is held.
            return trimmed == "|"
        }
        // A header that has no delimiter row under it yet.
        return true
    }

    /// A line that would be a table's header if a delimiter row followed:
    /// it starts with a pipe and the line above it is not a table row.
    private static func isTableHeader(_ line: Substring, before: Substring?) -> Bool {
        guard line.trimmingCharacters(in: .whitespaces).hasPrefix("|") else { return false }
        return !(before?.trimmingCharacters(in: .whitespaces).hasPrefix("|") ?? false)
    }

    private final class Parsed {
        let blocks: [MarkdownBlock]
        init(_ blocks: [MarkdownBlock]) { self.blocks = blocks }
    }

    /// The message list redraws every visible reply whenever one of them
    /// streams; finished replies should not be parsed again each time.
    private static let cache: NSCache<NSString, Parsed> = {
        let cache = NSCache<NSString, Parsed>()
        cache.countLimit = 256
        return cache
    }()

    /// The previous streaming parse. Only one reply streams at a time.
    private struct StreamMemo {
        var source: String
        /// The blocks before the last two.
        var closedBlocks: [MarkdownBlock]
        /// Where the second-to-last block starts in `source`, in UTF-8
        /// bytes.
        var openStart: Int
    }

    @MainActor private static var streamMemo: StreamMemo?

    /// The source's lines, each with where it starts (in UTF-8 bytes, so
    /// a streaming parse can find its place again). A `\r` before a line
    /// break is dropped, and leading tabs are expanded.
    private static func lines(of source: String) -> [(text: String, start: Int)] {
        var result: [(text: String, start: Int)] = []
        var start = 0
        for piece in source.split(separator: "\n", omittingEmptySubsequences: false) {
            var line = String(piece)
            let length = line.utf8.count
            if line.hasSuffix("\r") { line.removeLast() }
            result.append((expandingLeadingTabs(line), start))
            start += length + 1
        }
        return result
    }

    /// Indentation decides list nesting, so a leading tab counts as the
    /// spaces up to the next multiple of four, as CommonMark has it.
    private static func expandingLeadingTabs(_ line: String) -> String {
        guard line.prefix(while: { $0 == " " || $0 == "\t" }).contains("\t") else { return line }
        var expanded = ""
        var column = 0
        var index = line.startIndex
        while index < line.endIndex, line[index] == " " || line[index] == "\t" {
            let width = line[index] == "\t" ? 4 - column % 4 : 1
            expanded += String(repeating: " ", count: width)
            column += width
            index = line.index(after: index)
        }
        return expanded + line[index...]
    }
}

private struct BlockParser {
    private let lines: [String]
    private var index = 0
    private var blocks: [MarkdownBlock] = []
    private var paragraph: [String] = []
    private var paragraphStart = 0
    /// The line each of `blocks` starts on.
    private(set) var startLines: [Int] = []

    init(lines: [String]) {
        self.lines = lines
    }

    mutating func parse() -> [MarkdownBlock] {
        while index < lines.count {
            let line = lines[index]
            let start = index
            if Line.isBlank(line) {
                flushParagraph()
                index += 1
            } else if let fence = Fence(opening: line) {
                flushParagraph()
                parseCode(fence)
                startLines.append(start)
            } else if let heading = Line.heading(line) {
                flushParagraph()
                blocks.append(.heading(level: heading.level, text: heading.text))
                startLines.append(start)
                index += 1
            } else if Line.isRule(line) {
                flushParagraph()
                blocks.append(.rule)
                startLines.append(start)
                index += 1
            } else if Line.isQuote(line) {
                flushParagraph()
                parseQuote()
                startLines.append(start)
            } else if ListMarker(line) != nil {
                flushParagraph()
                parseList()
                startLines.append(start)
            } else if let alignments = tableAlignments(headerAt: index) {
                flushParagraph()
                parseTable(alignments: alignments)
                startLines.append(start)
            } else {
                if paragraph.isEmpty { paragraphStart = index }
                paragraph.append(line.trimmingCharacters(in: .whitespaces))
                index += 1
            }
        }
        flushParagraph()
        return blocks
    }

    private mutating func flushParagraph() {
        guard !paragraph.isEmpty else { return }
        blocks.append(.paragraph(paragraph.joined(separator: "\n")))
        startLines.append(paragraphStart)
        paragraph.removeAll()
    }

    private func nextNonBlank(from start: Int) -> Int? {
        var next = start
        while next < lines.count, Line.isBlank(lines[next]) { next += 1 }
        return next < lines.count ? next : nil
    }

    // MARK: - Code

    private mutating func parseCode(_ fence: Fence) {
        index += 1
        var body: [String] = []
        while index < lines.count {
            let line = lines[index]
            index += 1
            if fence.isClosed(by: line) { break }
            // The first one or two backticks of the closing fence, streamed
            // ahead of the rest, would flash as a line of code.
            if index == lines.count, fence.isPartialClose(line) { break }
            body.append(fence.strippingIndent(line))
        }
        blocks.append(.code(language: fence.language, code: body.joined(separator: "\n")))
    }

    // MARK: - Quotes

    private mutating func parseQuote() {
        var inner: [String] = []
        while index < lines.count, Line.isQuote(lines[index]) {
            inner.append(Line.droppingQuoteMarker(lines[index]))
            index += 1
        }
        var nested = BlockParser(lines: inner)
        blocks.append(.quote(nested.parse()))
    }

    // MARK: - Lists

    private mutating func parseList() {
        guard let first = ListMarker(lines[index]) else { return }
        let base = first.indent
        var items: [MarkdownListItem] = []

        while index < lines.count {
            if Line.isBlank(lines[index]) {
                // A loose list: blank lines between items keep it going.
                guard let next = nextNonBlank(from: index),
                      let marker = ListMarker(lines[next]),
                      marker.isOrdered == first.isOrdered,
                      marker.indent <= base + 1
                else { break }
                index = next
                continue
            }
            guard let marker = ListMarker(lines[index]),
                  marker.isOrdered == first.isOrdered,
                  marker.indent <= base + 1
            else { break }
            index += 1
            items.append(parseItem(marker, base: base))
        }
        blocks.append(.list(MarkdownList(isOrdered: first.isOrdered, start: first.number, items: items)))
    }

    /// Everything indented past the list's own markers belongs to the
    /// item: nested lists, code, further paragraphs. Those lines are
    /// dedented and parsed as a document of their own.
    private mutating func parseItem(_ marker: ListMarker, base: Int) -> MarkdownListItem {
        var text = marker.content
        var childLines: [String] = []
        var childIndent: Int?
        var sawBlank = false
        var openFence: Fence?

        while index < lines.count {
            let line = lines[index]

            if let fence = openFence {
                // Inside a fence every line is code until it closes, however
                // it is indented; models often leave code in a list item
                // unindented.
                childLines.append(Line.droppingIndent(line, upTo: childIndent ?? 0))
                if fence.isClosed(by: line) { openFence = nil }
                index += 1
                continue
            }

            if Line.isBlank(line) {
                guard let next = nextNonBlank(from: index),
                      Line.indentation(lines[next]) > base,
                      ListMarker(lines[next]).map({ $0.indent > base + 1 }) ?? true
                else { break }
                childLines.append(contentsOf: repeatElement("", count: next - index))
                index = next
                sawBlank = true
                continue
            }

            if let sibling = ListMarker(line), sibling.indent <= base + 1 { break }

            let continuesText = childLines.isEmpty && !sawBlank && !Line.startsBlock(line)
            if Line.indentation(line) > base {
                if continuesText {
                    text += "\n" + line.trimmingCharacters(in: .whitespaces)
                } else {
                    let strip = childIndent ?? min(Line.indentation(line), marker.contentOffset)
                    childIndent = strip
                    let dedented = Line.droppingIndent(line, upTo: strip)
                    childLines.append(dedented)
                    if let fence = Fence(opening: dedented) { openFence = fence }
                }
                index += 1
                continue
            }

            // Unindented text right after the item still belongs to its
            // paragraph (CommonMark's lazy continuation); anything else
            // ends the list.
            guard continuesText else { break }
            text += "\n" + line.trimmingCharacters(in: .whitespaces)
            index += 1
        }

        var nested = BlockParser(lines: childLines)
        return MarkdownListItem(text: text, children: nested.parse())
    }

    // MARK: - Tables

    private func tableAlignments(headerAt row: Int) -> [MarkdownTable.Alignment]? {
        guard row + 1 < lines.count, lines[row].contains("|") else { return nil }
        let columns = Line.tableCells(lines[row]).count
        return Line.tableAlignments(lines[row + 1], columns: columns)
    }

    private mutating func parseTable(alignments: [MarkdownTable.Alignment]) {
        let header = Line.tableCells(lines[index])
        index += 2
        var rows: [[String]] = []
        while index < lines.count, !Line.isBlank(lines[index]), lines[index].contains("|") {
            rows.append(Line.padded(Line.tableCells(lines[index]), to: header.count))
            index += 1
        }
        blocks.append(.table(MarkdownTable(header: header, alignments: alignments, rows: rows)))
    }
}

/// An opening code fence: three or more backticks or tildes, then an
/// optional language.
private struct Fence {
    let indent: Int
    let marker: Character
    let length: Int
    let language: String

    init?(opening line: String) {
        let indent = Line.indentation(line)
        guard indent <= 3 else { return nil }
        let rest = line.dropFirst(indent)
        guard let marker = rest.first, marker == "`" || marker == "~" else { return nil }
        let length = rest.prefix(while: { $0 == marker }).count
        guard length >= 3 else { return nil }
        let info = rest.dropFirst(length).trimmingCharacters(in: .whitespaces)
        if marker == "`", info.contains("`") { return nil }
        self.indent = indent
        self.marker = marker
        self.length = length
        self.language = info.split(separator: " ").first.map(String.init) ?? ""
    }

    func isClosed(by line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        return trimmed.count >= length && trimmed.allSatisfy { $0 == marker }
    }

    func isPartialClose(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        return !trimmed.isEmpty && trimmed.count < length && trimmed.allSatisfy { $0 == marker }
    }

    /// Content lines lose up to the fence's own indentation.
    func strippingIndent(_ line: String) -> String {
        Line.droppingIndent(line, upTo: indent)
    }
}

/// A bullet (`-`, `*`, `+`) or ordered (`1.`, `1)`) list marker.
private struct ListMarker {
    let indent: Int
    let isOrdered: Bool
    let number: Int
    let content: String
    /// The column the item's text starts at; nested content is indented
    /// to about here.
    let contentOffset: Int

    init?(_ line: String) {
        let chars = Array(line)
        var position = 0
        while position < chars.count, chars[position] == " " { position += 1 }
        guard position < chars.count else { return nil }
        let indent = position
        var isOrdered = false
        var number = 0

        if chars[position] == "-" || chars[position] == "*" || chars[position] == "+" {
            position += 1
        } else {
            var end = position
            while end < chars.count, end - position < 9, chars[end].isASCII, chars[end].isNumber { end += 1 }
            guard end > position, end < chars.count, chars[end] == "." || chars[end] == ")" else { return nil }
            number = Int(String(chars[position..<end])) ?? 1
            isOrdered = true
            position = end + 1
        }

        // The marker is followed by a space, or ends the line (an item
        // whose text has not streamed in yet).
        if position < chars.count, chars[position] != " " { return nil }
        var textStart = position
        while textStart < chars.count, chars[textStart] == " " { textStart += 1 }

        self.indent = indent
        self.isOrdered = isOrdered
        self.number = number
        self.content = String(chars[textStart...]).trimmingCharacters(in: .whitespaces)
        // Five or more spaces after a marker would start indented code in
        // CommonMark; a chat model never means that.
        self.contentOffset = textStart - position > 4 ? position + 1 : max(textStart, position + 1)
    }
}

private enum Line {
    static func isBlank(_ line: String) -> Bool {
        line.allSatisfy { $0 == " " || $0 == "\t" }
    }

    static func indentation(_ line: String) -> Int {
        line.prefix(while: { $0 == " " }).count
    }

    static func droppingIndent(_ line: String, upTo count: Int) -> String {
        String(line.dropFirst(min(count, indentation(line))))
    }

    /// A line that starts a block of its own rather than continuing the
    /// paragraph above it.
    static func startsBlock(_ line: String) -> Bool {
        Fence(opening: line) != nil || heading(line) != nil || isRule(line)
            || isQuote(line) || ListMarker(line) != nil
    }

    static func heading(_ line: String) -> (level: Int, text: String)? {
        guard indentation(line) <= 3 else { return nil }
        let rest = line.drop(while: { $0 == " " })
        let level = rest.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(level) else { return nil }
        let after = rest.dropFirst(level)
        guard after.isEmpty || after.first == " " else { return nil }
        var text = after.trimmingCharacters(in: .whitespaces)
        if text.allSatisfy({ $0 == "#" }) {
            text = ""
        } else if let closing = text.range(of: #"\s+#+$"#, options: .regularExpression) {
            text.removeSubrange(closing)
        }
        return (level, text)
    }

    static func isRule(_ line: String) -> Bool {
        guard indentation(line) <= 3 else { return false }
        let marks = line.filter { $0 != " " }
        guard marks.count >= 3, let first = marks.first, "-*_".contains(first) else { return false }
        return marks.allSatisfy { $0 == first }
    }

    static func isQuote(_ line: String) -> Bool {
        indentation(line) <= 3 && line.drop(while: { $0 == " " }).first == ">"
    }

    static func droppingQuoteMarker(_ line: String) -> String {
        var rest = line.drop(while: { $0 == " " }).dropFirst()
        if rest.first == " " { rest = rest.dropFirst() }
        return String(rest)
    }

    /// A row's cells, split on pipes that are not escaped.
    static func tableCells(_ line: String) -> [String] {
        var row = line.trimmingCharacters(in: .whitespaces)
        if row.hasPrefix("|") { row.removeFirst() }
        if row.hasSuffix("|"), !row.hasSuffix("\\|") { row.removeLast() }
        var cells: [String] = []
        var current = ""
        var previous: Character?
        for char in row {
            if char == "|", previous != "\\" {
                cells.append(current)
                current = ""
            } else {
                current.append(char)
            }
            previous = char
        }
        cells.append(current)
        return cells.map {
            $0.replacingOccurrences(of: "\\|", with: "|").trimmingCharacters(in: .whitespaces)
        }
    }

    /// The column alignments a delimiter row (`| :--- | ---: |`) sets, or
    /// nil if the line is not one. A row with fewer cells than the header
    /// is accepted so a table streaming in is drawn as a table at once.
    static func tableAlignments(_ line: String, columns: Int) -> [MarkdownTable.Alignment]? {
        guard line.contains("|"), line.contains("-") else { return nil }
        let cells = tableCells(line)
        guard cells.count <= columns else { return nil }
        var alignments: [MarkdownTable.Alignment] = []
        for cell in cells {
            guard cell.range(of: #"^:?-+:?$"#, options: .regularExpression) != nil else { return nil }
            switch (cell.hasPrefix(":"), cell.hasSuffix(":")) {
            case (true, true): alignments.append(.center)
            case (false, true): alignments.append(.trailing)
            default: alignments.append(.leading)
            }
        }
        return alignments + Array(repeating: .leading, count: columns - alignments.count)
    }

    /// The beginning of a table's delimiter row before its first dash: only
    /// pipes, colons and spaces.
    static func isBareDelimiterStart(_ line: Substring) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        return trimmed.hasPrefix("|") && trimmed.allSatisfy { "|: ".contains($0) }
    }

    static func padded(_ cells: [String], to count: Int) -> [String] {
        Array((cells + Array(repeating: "", count: max(0, count - cells.count))).prefix(count))
    }
}
