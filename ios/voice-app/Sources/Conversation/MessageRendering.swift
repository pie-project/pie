import Foundation

/// How stored messages become text for something other than the chat
/// screen: the model's prompt, the speech synthesizer, the share sheet and
/// the sidebar title.
enum MessageRendering {

    /// Shown in place of a reply that generated nothing. Never sent back to
    /// the model as if it had said it.
    static let noReplyPlaceholder = "(no reply)"

    /// Shown in place of a reply the engine failed to produce, above the
    /// row's actions, so Regenerate is there to try again. Never sent back
    /// to the model or read aloud.
    static let failedReplyPlaceholder = "Something went wrong while generating this reply."

    /// One of the stand-ins above rather than anything the model said.
    static func isPlaceholder(_ text: String) -> Bool {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return text == noReplyPlaceholder || text == failedReplyPlaceholder
    }

    // MARK: - Prompt

    /// The transcript the backend is handed to answer the last message of
    /// `messages`, which is the user message being answered.
    ///
    /// Earlier turns go in as whole exchanges, and only exchanges whose
    /// reply has text: a question whose reply failed, was stopped before a
    /// word, or produced nothing would otherwise sit in the history
    /// unanswered, and a small model tends to answer it again instead of
    /// the new one.
    static func prompt(system: String, messages: [StoredMessage]) -> [PromptMessage] {
        var prompt = [PromptMessage(role: .system, content: system)]
        guard let question = messages.last, question.role == .user else { return prompt }

        let history = Array(messages.dropLast())
        var index = 0
        while index < history.count {
            let message = history[index]
            if message.role == .user,
               index + 1 < history.count,
               history[index + 1].role == .assistant,
               isUsableReply(history[index + 1]) {
                prompt.append(PromptMessage(role: .user, content: content(of: message)))
                prompt.append(PromptMessage(role: .assistant, content: history[index + 1].text))
                index += 2
            } else {
                index += 1
            }
        }
        prompt.append(PromptMessage(role: .user, content: content(of: question)))
        return prompt
    }

    /// What the model reads for a user message: each attachment's text in
    /// a labelled block, then what the user typed.
    ///
    /// The attachments share `PieRuntimeConfig.attachmentTokensPerMessage`,
    /// however many there are, so a message with several long documents
    /// still fits the context with room for the reply.
    static func content(of message: StoredMessage) -> String {
        let sizes = message.attachments.map { PieRuntimeConfig.estimatedTokens(in: $0.extractedText) }
        let budgets = attachmentBudgets(for: sizes)
        var parts = message.attachments.indices.map { index in
            render(message.attachments[index], size: sizes[index], budget: budgets[index])
        }
        let text = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            parts.append(message.viaVoice ? spokenQuestion(text) : text)
        }
        return parts.joined(separator: "\n\n")
    }

    /// A question asked in voice mode, as the model reads it.
    static func spokenQuestion(_ text: String) -> String {
        text + "\n\n" + PieRuntimeConfig.spokenReplyInstruction
    }

    /// Each attachment's share of the message's budget, in estimated
    /// tokens, given each one's size: short ones are shown whole and the
    /// long ones split what is left evenly, none past
    /// `PieRuntimeConfig.attachmentTokensEach`.
    static func attachmentBudgets(for sizes: [Int]) -> [Int] {
        var budgets = Array(repeating: 0, count: sizes.count)
        var remaining = PieRuntimeConfig.attachmentTokensPerMessage
        let smallestFirst = sizes.indices.sorted { sizes[$0] < sizes[$1] }
        for (position, index) in smallestFirst.enumerated() {
            let share = remaining / (smallestFirst.count - position)
            budgets[index] = min(sizes[index], share, PieRuntimeConfig.attachmentTokensEach)
            remaining -= budgets[index]
        }
        return budgets
    }

    private static func render(_ attachment: Attachment, size: Int, budget: Int) -> String {
        var body = attachment.extractedText
        if size > budget {
            body = PieRuntimeConfig.prefix(of: body, tokens: budget) + "\n[The rest of this attachment was left out.]"
        }
        switch attachment.kind {
        case .photo:
            return "[Attached photo: text recognised on this iPhone]\n\(body)\n[End of attachment]"
        case .file:
            return "[Attached file \(attachment.name)]\n\(body)\n[End of attachment]"
        }
    }

    private static func isUsableReply(_ message: StoredMessage) -> Bool {
        let text = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
        return !message.isStreaming && !text.isEmpty && !isPlaceholder(text)
    }

    // MARK: - Speech

    /// `markdown` as a synthesizer should say it: no heading marks,
    /// emphasis, backticks, link targets, table rules or emoji, and each
    /// code block replaced by a short notice, since reading code aloud
    /// symbol by symbol helps nobody. Lines stay on lines so the sentence
    /// chunker treats each list item as its own utterance.
    static func speakable(_ markdown: String) -> String {
        var lines: [String] = []
        var inCodeBlock = false
        for raw in markdown.components(separatedBy: .newlines) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("```") || line.hasPrefix("~~~") {
                if !inCodeBlock { lines.append("Code block omitted.") }
                inCodeBlock.toggle()
                continue
            }
            if inCodeBlock { continue }
            let spoken = speakableLine(line)
            if !spoken.isEmpty { lines.append(spoken) }
        }
        return lines.joined(separator: "\n")
    }

    private static func speakableLine(_ line: String) -> String {
        // Horizontal rules and table separator rows say nothing.
        if line.range(of: #"^([-*_]\s*){3,}$"#, options: .regularExpression) != nil { return "" }
        if line.range(of: #"^\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*$"#, options: .regularExpression) != nil { return "" }

        var text = line.filter { !isPictograph($0) }
        text = text.replacingOccurrences(of: #"^#{1,6}\s*"#, with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: #"^>\s?"#, with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: #"^[-*+]\s+"#, with: "", options: .regularExpression)
        // Links and images keep their words and lose their targets.
        text = text.replacingOccurrences(of: #"!?\[([^\]]*)\]\([^)]*\)"#, with: "$1", options: .regularExpression)
        // Table cells read as a list.
        if text.hasPrefix("|") {
            text = text
                .split(separator: "|")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
                .joined(separator: ", ")
        }
        for mark in ["**", "__", "*", "`", "#"] {
            text = text.replacingOccurrences(of: mark, with: "")
        }
        return text.trimmingCharacters(in: .whitespaces)
    }

    /// An emoji, which a synthesizer reads out by name ("smiling face with
    /// smiling eyes"). Digits, "#", "*", "©" and "™" carry the emoji
    /// property too but are drawn as text, so they stay, and so does the
    /// joiner Indic scripts put between letters.
    private static func isPictograph(_ character: Character) -> Bool {
        let scalars = character.unicodeScalars
        if scalars.contains(where: { $0.properties.isEmojiPresentation || $0.properties.isEmojiModifier }) {
            return true
        }
        // A symbol that is text by default, asked to be drawn as a picture
        // ("❤️") or joined into one ("🏳️‍🌈"). A keycap ("1️⃣") is built
        // on a digit, which is worth saying.
        let drawnAsPicture = scalars.contains("\u{FE0F}") || scalars.contains("\u{200D}")
        return drawnAsPicture && scalars.contains { $0.properties.isEmoji && !$0.isASCII }
    }

    // MARK: - Share sheet

    /// The conversation as Markdown: the title, then each message under
    /// who said it.
    static func shareText(_ conversation: Conversation) -> String {
        var blocks = ["# \(conversation.displayTitle)"]
        for message in conversation.messages where message.role != .system {
            let text = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
            switch message.role {
            case .user:
                var block = "**You:** \(text)"
                if !message.attachments.isEmpty {
                    let names = message.attachments.map(\.name).joined(separator: ", ")
                    block += text.isEmpty ? "_Attached: \(names)_" : "\n\n_Attached: \(names)_"
                }
                blocks.append(block)
            case .assistant:
                guard !text.isEmpty, !isPlaceholder(text) else { continue }
                blocks.append("**Pie:** \(text)")
            case .system:
                continue
            }
        }
        return blocks.joined(separator: "\n\n")
    }

    // MARK: - Titles

    /// The longest title kept for the sidebar.
    static let titleLimit = 40

    /// A model's title reply cut down to the title itself: first line only,
    /// no "Title:" lead-in, no quotes or emphasis, no trailing punctuation,
    /// at most `titleLimit` characters. Nil when nothing is left.
    static func cleanTitle(_ raw: String) -> String? {
        let firstLine = raw
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty } ?? ""

        var title = firstLine.replacingOccurrences(
            of: #"^(title)\s*:\s*"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        for mark in ["\"", "\u{201C}", "\u{201D}", "\u{00AB}", "\u{00BB}", "**", "*", "`", "#"] {
            title = title.replacingOccurrences(of: mark, with: "")
        }
        // Single quotes only at the ends: inside, they are apostrophes.
        // Quotes and punctuation can interleave at the end ("'Hi'."), so
        // the end is trimmed of both together.
        let leading = CharacterSet(charactersIn: "'\u{2018}\u{2019} ")
        let trailing = CharacterSet(charactersIn: "'\u{2018}\u{2019}.,;:!?\u{2026}- ")
        while let first = title.unicodeScalars.first, leading.contains(first) {
            title.removeFirst()
        }
        while let last = title.unicodeScalars.last, trailing.contains(last) {
            title.removeLast()
        }

        if title.count > titleLimit {
            let clipped = String(title.prefix(titleLimit))
            // Prefer ending on a whole word.
            if let space = clipped.lastIndex(of: " "), clipped.distance(from: clipped.startIndex, to: space) > titleLimit / 2 {
                title = String(clipped[..<space])
            } else {
                title = clipped
            }
        }
        title = title.trimmingCharacters(in: .whitespaces)
        return title.isEmpty ? nil : title
    }
}
