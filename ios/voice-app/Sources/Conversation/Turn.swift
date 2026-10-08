import Foundation

/// One message of the transcript as the backend sees it. The backend is
/// stateless across turns, so the controller sends the conversation so
/// far with every utterance; this is the unit it sends.
struct ChatMessage: Equatable {
    enum Role: String {
        case user
        case assistant
    }

    let role: Role
    let content: String
}

struct Turn: Identifiable, Equatable {
    enum Speaker {
        case user
        case assistant
    }

    let id = UUID()
    let speaker: Speaker
    var text: String
    var stats: TurnStats?
    /// Still being generated — the view shows a caret while true.
    var isStreaming: Bool = false

    /// This turn, as the line of transcript the backend is handed.
    var message: ChatMessage {
        ChatMessage(role: speaker == .user ? .user : .assistant, content: text)
    }

    static func == (lhs: Turn, rhs: Turn) -> Bool {
        lhs.id == rhs.id && lhs.text == rhs.text && lhs.isStreaming == rhs.isStreaming
    }
}
