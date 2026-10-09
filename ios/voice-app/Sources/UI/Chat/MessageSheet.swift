import Foundation

/// The sheets a message's context menu opens. The message list presents
/// them, not the rows, because a row in a lazy stack can be torn down
/// while its sheet is still up.
enum MessageSheet: Identifiable {
    case edit(StoredMessage)
    case selectText(String)

    var id: String {
        switch self {
        case .edit(let message): return "edit-\(message.id)"
        case .selectText(let text): return "select-\(text.hashValue)"
        }
    }
}
