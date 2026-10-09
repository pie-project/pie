import Foundation

/// Saved conversations: one JSON file per thread under
/// Application Support/PieVoice/chats/. Temporary conversations are never
/// written.
///
/// One file per conversation rather than one file for all of them: a save
/// after every reply then rewrites a single small file, and a file that
/// cannot be read costs one conversation instead of the whole history.
@MainActor
final class ChatStore: ObservableObject {

    /// Saved (non-temporary) conversations, most recently updated first.
    @Published private(set) var conversations: [Conversation] = []

    private let directory: URL?

    init() {
        directory = Self.makeDirectory()
        conversations = Self.loadAll(from: directory)
    }

    func conversation(_ id: UUID) -> Conversation? {
        conversations.first { $0.id == id }
    }

    /// Inserts or replaces; ignores temporary and empty conversations.
    func save(_ conversation: Conversation) {
        guard !conversation.isTemporary, !conversation.isEmpty else { return }
        var stored = conversation
        // A reply still streaming is saved as it stands, marked stopped:
        // the finished reply overwrites this copy, so it only survives if
        // the app is killed mid-reply, and then the reply was cut off. The
        // row it comes back as offers Regenerate.
        for index in stored.messages.indices where stored.messages[index].isStreaming {
            stored.messages[index].isStreaming = false
            stored.messages[index].wasStopped = true
        }
        conversations.removeAll { $0.id == stored.id }
        conversations.append(stored)
        sort()
        write(stored)
    }

    func rename(_ id: UUID, to title: String) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        conversations[index].title = trimmed
        write(conversations[index])
    }

    func delete(_ id: UUID) {
        conversations.removeAll { $0.id == id }
        if let url = fileURL(for: id) {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Removes every conversation file, including any that could not be
    /// read (an older format, a partial write): those are not listed, but
    /// they are still the user's chats, and this is the button that
    /// promises none are left on the phone.
    func deleteAll() {
        conversations = []
        guard
            let directory,
            let files = try? FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil
            )
        else { return }
        for file in files where file.pathExtension == "json" {
            try? FileManager.default.removeItem(at: file)
        }
    }

    /// Title and message text, case-insensitive; empty query = all.
    func search(_ query: String) -> [Conversation] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return conversations }
        return conversations.filter { conversation in
            conversation.displayTitle.localizedCaseInsensitiveContains(needle)
                || conversation.messages.contains { $0.text.localizedCaseInsensitiveContains(needle) }
        }
    }

    /// Sidebar sections in order: "Today", "Yesterday", "Previous 7 Days",
    /// "Previous 30 Days", then one per older month ("September 2026").
    /// Each section keeps the most recently updated first; empty sections
    /// are left out.
    static func grouped(_ conversations: [Conversation], now: Date = Date())
        -> [(title: String, items: [Conversation])] {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: now)
        let monthFormatter = DateFormatter()
        monthFormatter.locale = Locale.current
        monthFormatter.setLocalizedDateFormatFromTemplate("MMMMyyyy")

        var sections: [(title: String, items: [Conversation])] = []
        func add(_ conversation: Conversation, to title: String) {
            if let index = sections.firstIndex(where: { $0.title == title }) {
                sections[index].items.append(conversation)
            } else {
                sections.append((title: title, items: [conversation]))
            }
        }

        // Walking newest first means sections are created in display
        // order, months included.
        for conversation in conversations.sorted(by: { $0.updatedAt > $1.updatedAt }) {
            let day = calendar.startOfDay(for: conversation.updatedAt)
            let daysAgo = calendar.dateComponents([.day], from: day, to: today).day ?? 0
            switch daysAgo {
            case ...0: add(conversation, to: "Today")
            case 1: add(conversation, to: "Yesterday")
            case 2...7: add(conversation, to: "Previous 7 Days")
            case 8...30: add(conversation, to: "Previous 30 Days")
            default: add(conversation, to: monthFormatter.string(from: conversation.updatedAt))
            }
        }
        return sections
    }

    // MARK: - Files

    private func sort() {
        conversations.sort { $0.updatedAt > $1.updatedAt }
    }

    private func fileURL(for id: UUID) -> URL? {
        directory?.appendingPathComponent(id.uuidString.lowercased() + ".json")
    }

    private func write(_ conversation: Conversation) {
        guard let url = fileURL(for: conversation.id) else { return }
        do {
            let data = try Self.encoder.encode(conversation)
            try data.write(to: url, options: .atomic)
        } catch {
            print("[store] could not save conversation \(conversation.id): \(error.localizedDescription)")
        }
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    private static func makeDirectory() -> URL? {
        do {
            let support = try FileManager.default.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            let chats = support
                .appendingPathComponent("PieVoice", isDirectory: true)
                .appendingPathComponent("chats", isDirectory: true)
            try FileManager.default.createDirectory(at: chats, withIntermediateDirectories: true)
            return chats
        } catch {
            // Without a directory the app still works; chats just do not
            // outlive the process.
            print("[store] no chats directory: \(error.localizedDescription)")
            return nil
        }
    }

    /// Every conversation file that decodes; one that does not (a partial
    /// write from an older build, a format change) is skipped rather than
    /// taking the rest of the history down with it.
    private static func loadAll(from directory: URL?) -> [Conversation] {
        guard
            let directory,
            let files = try? FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
        else { return [] }

        var loaded: [Conversation] = []
        for file in files where file.pathExtension == "json" {
            guard
                let data = try? Data(contentsOf: file),
                let conversation = try? decoder.decode(Conversation.self, from: data),
                !conversation.isTemporary
            else {
                print("[store] skipped unreadable \(file.lastPathComponent)")
                continue
            }
            loaded.append(conversation)
        }
        return loaded.sorted { $0.updatedAt > $1.updatedAt }
    }
}
