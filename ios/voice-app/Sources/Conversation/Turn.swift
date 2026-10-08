import Foundation

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

    static func == (lhs: Turn, rhs: Turn) -> Bool {
        lhs.id == rhs.id && lhs.text == rhs.text && lhs.isStreaming == rhs.isStreaming
    }
}
