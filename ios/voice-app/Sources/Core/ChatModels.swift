import Foundation

/// Who said a message.
enum ChatRole: String, Codable, Equatable {
    case system
    case user
    case assistant
}

/// One message as the backend is handed it: the rendered transcript the
/// model reads, nothing about how it is shown.
struct PromptMessage: Equatable {
    let role: ChatRole
    let content: String
}

/// What one generated reply cost, as the engine accounts for it.
///
/// `reused` is the point of serving through Pie: prompt tokens the engine
/// served from the KV pages and recurrent state it kept from earlier
/// turns, so they did not have to be prefilled again.
struct TurnStats: Codable, Equatable {
    /// Everything the model was shown this turn.
    var promptTokens: Int = 0
    var reused: Int = 0
    var newPrefill: Int = 0
    var generated: Int = 0
    var resumed: Bool = false
    /// Request to return, in seconds.
    var elapsed: TimeInterval = 0
    /// Request to the first streamed event, reply or reasoning text.
    var timeToFirstToken: TimeInterval?
    /// Set when the backend fell back to a slower path for this turn.
    var note: String = ""

    /// Decode rate over the generation window, excluding the time to the
    /// first token so prompt length does not distort it.
    var tokensPerSecond: Double {
        let window = elapsed - (timeToFirstToken ?? 0)
        guard window > 0, generated > 1 else { return 0 }
        return Double(generated - 1) / window
    }
}

/// The thumbs on an assistant message.
enum Feedback: String, Codable, Equatable {
    case good
    case bad
}

/// Something the user attached to a message. The model is text-only, so
/// what it reads is the text extracted on the device (recognised in a
/// photo, or read out of a document); the thumbnail is for the screen.
struct Attachment: Codable, Identifiable, Equatable {
    enum Kind: String, Codable, Equatable {
        case photo
        case file
    }

    let id: UUID
    var kind: Kind
    /// A file's name, or "Photo".
    var name: String
    /// What the model is shown for this attachment.
    var extractedText: String
    /// A small JPEG preview of a photo; nil for files.
    var thumbnailJPEG: Data?

    init(
        id: UUID = UUID(),
        kind: Kind,
        name: String,
        extractedText: String,
        thumbnailJPEG: Data? = nil
    ) {
        self.id = id
        self.kind = kind
        self.name = name
        self.extractedText = extractedText
        self.thumbnailJPEG = thumbnailJPEG
    }
}

/// One message of a conversation as the app keeps it.
struct StoredMessage: Codable, Identifiable, Equatable {
    let id: UUID
    var role: ChatRole
    var text: String
    /// What the model reasoned before answering, in thinking mode.
    var reasoning: String
    /// How long the reasoning took, for the "Thought for Ns" disclosure.
    var thoughtSeconds: Double?
    var attachments: [Attachment]
    var stats: TurnStats?
    var feedback: Feedback?
    /// The user stopped this reply before it finished; the text is partial.
    var wasStopped: Bool
    /// Asked or answered in voice mode.
    var viaVoice: Bool
    var createdAt: Date
    /// Still being generated. Never true in a saved file.
    var isStreaming: Bool

    init(
        id: UUID = UUID(),
        role: ChatRole,
        text: String,
        reasoning: String = "",
        thoughtSeconds: Double? = nil,
        attachments: [Attachment] = [],
        stats: TurnStats? = nil,
        feedback: Feedback? = nil,
        wasStopped: Bool = false,
        viaVoice: Bool = false,
        createdAt: Date = Date(),
        isStreaming: Bool = false
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.reasoning = reasoning
        self.thoughtSeconds = thoughtSeconds
        self.attachments = attachments
        self.stats = stats
        self.feedback = feedback
        self.wasStopped = wasStopped
        self.viaVoice = viaVoice
        self.createdAt = createdAt
        self.isStreaming = isStreaming
    }
}

/// A chat thread.
struct Conversation: Codable, Identifiable, Equatable {
    let id: UUID
    var title: String
    var messages: [StoredMessage]
    var createdAt: Date
    var updatedAt: Date
    /// Never written to disk and never listed in the sidebar.
    var isTemporary: Bool

    init(
        id: UUID = UUID(),
        title: String = "",
        messages: [StoredMessage] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        isTemporary: Bool = false
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.isTemporary = isTemporary
    }

    /// The prefix-cache namespace this conversation's turns publish to and
    /// adopt from inside the engine.
    var sessionKey: String { "chat-" + id.uuidString.lowercased() }

    var isEmpty: Bool { messages.isEmpty }

    /// What the sidebar shows before a title has been generated.
    var displayTitle: String {
        if !title.isEmpty { return title }
        let first = messages.first(where: { $0.role == .user })?.text ?? ""
        let line = first.split(whereSeparator: \.isNewline).first.map(String.init) ?? ""
        return line.isEmpty ? "New chat" : String(line.prefix(48))
    }
}

/// How hard the model works on a reply. Mirrors ChatGPT's Instant and
/// Thinking: thinking lets the model reason before it answers.
enum ReplyMode: String, Codable, CaseIterable, Identifiable {
    case instant
    case thinking

    var id: String { rawValue }

    var title: String {
        switch self {
        case .instant: return "Instant"
        case .thinking: return "Thinking"
        }
    }

    var subtitle: String {
        switch self {
        case .instant: return "Answers right away"
        case .thinking: return "Thinks first for better answers"
        }
    }
}

/// Generation settings for one reply.
struct ReplyOptions: Equatable {
    var maxTokens: Int
    var temperature: Double
    var topP: Double
    var think: Bool
}

/// Text as it streams out of the engine.
enum ReplyEvent: Equatable {
    /// Reply text, for the screen and the speaker.
    case text(String)
    /// Reasoning text, in thinking mode. Never spoken.
    case reasoning(String)
}

/// How a reply ended.
struct ReplyResult: Equatable {
    var text: String
    var reasoning: String
    var stats: TurnStats
    /// Stopped by `ConversationBackend.cancel`; `text` is what was
    /// produced before the stop.
    var cancelled: Bool
}

/// Names one reply so it can be cancelled, including before it starts.
/// Identifiers are unique and increasing for the life of the process.
final class ReplyTicket: @unchecked Sendable {
    let id: UInt64

    init() {
        id = ReplyTicket.nextID()
    }

    private static let lock = NSLock()
    private static var counter: UInt64 = 0

    private static func nextID() -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        counter += 1
        return counter
    }
}

enum ConversationError: LocalizedError {
    case backend(String)

    var errorDescription: String? {
        switch self {
        case .backend(let message): return message
        }
    }
}
