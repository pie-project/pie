import Foundation
import SwiftUI
import UIKit

/// Saved conversations: one JSON file per thread under
/// Application Support/PieVoice/chats/. Temporary conversations are never
/// written.
///
/// One file per conversation rather than one file for all of them: a save
/// after every reply then rewrites a single small file, and a file that
/// cannot be read costs one conversation instead of the whole history.
///
/// The list changes at once, on the main thread, and animated, so the
/// sidebar slides a new chat in at the top, crossfades a rename and
/// collapses a deleted row. The files follow on a background queue, in
/// order: a save never costs the frame in which a send, a new chat or a
/// switch starts its animation.
///
/// Off the main thread, a write could still be waiting when the app is
/// put away, and a suspended app writes nothing: a reply finished just as
/// the user swiped home, then quit from the app switcher, was lost. So
/// each write asks iOS for the moment it needs to finish in the background
/// (`beginBackgroundTask`), and the queue is drained when the app is told
/// it is about to end.
@MainActor
final class ChatStore: ObservableObject {

    /// Saved (non-temporary) conversations, most recently updated first.
    @Published private(set) var conversations: [Conversation] = [] {
        didSet { history = nil }
    }

    private let directory: URL?

    /// The sidebar's sections for `conversations`, worked out once per
    /// change to the list and per day ("Today" moves at midnight) rather
    /// than on every redraw of the sidebar.
    private var history: (day: Date, sections: [(title: String, items: [Conversation])])?

    /// File writes and deletes, one at a time and in the order they were
    /// asked for, so a delete can never be overtaken by an older write
    /// that would bring the chat back.
    private static let disk = DispatchQueue(label: "PieVoice.ChatStore.disk", qos: .utility)

    private var terminationObserver: NSObjectProtocol?

    init() {
        directory = Self.makeDirectory()
        conversations = Self.loadAll(from: directory)
        // Posted on the main thread; the app has a few seconds left.
        terminationObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.willTerminateNotification, object: nil, queue: .main
        ) { _ in
            MainActor.assumeIsolated { Self.finishPendingWrites() }
        }
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
        var updated = conversations.filter { $0.id != stored.id }
        updated.append(stored)
        updated.sort { $0.updatedAt > $1.updatedAt }
        publish(updated)
        write(stored)
    }

    func rename(_ id: UUID, to title: String) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var updated = conversations
        updated[index].title = trimmed
        publish(updated)
        write(updated[index])
    }

    func delete(_ id: UUID) {
        publish(conversations.filter { $0.id != id })
        guard let url = fileURL(for: id) else { return }
        Self.onDisk("Delete chat") {
            ChatFiles.remove(url)
        }
    }

    /// Removes every conversation file, including any that could not be
    /// read (an older format, a partial write): those are not listed, but
    /// they are still the user's chats, and this is the button that
    /// promises none are left on the phone.
    func deleteAll() {
        publish([])
        guard let directory else { return }
        Self.onDisk("Delete all chats") {
            ChatFiles.removeAll(in: directory)
        }
    }

    /// Waits until every write and delete asked for so far has reached the
    /// disk. For quitting (on purpose, to load another model, or because
    /// iOS is ending the app), so a reply saved a moment before is not lost.
    static func finishPendingWrites() {
        disk.sync {}
    }

    /// Runs `work` on the disk queue, holding a background task until it is
    /// done, so that going to the home screen cannot suspend the app with
    /// the file half way. (The queue is serial: earlier work finishes
    /// first, under its own task.)
    private static func onDisk(_ name: String, _ work: @escaping @Sendable () -> Void) {
        let task = UIApplication.shared.beginBackgroundTask(withName: name)
        disk.async {
            work()
            Task { @MainActor in
                if task != .invalid { UIApplication.shared.endBackgroundTask(task) }
            }
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

    /// `grouped(conversations)`, kept until the list or the day changes.
    func groupedHistory(now: Date = Date()) -> [(title: String, items: [Conversation])] {
        let day = Calendar.current.startOfDay(for: now)
        if let history, history.day == day { return history.sections }
        let sections = Self.grouped(conversations, now: now)
        history = (day: day, sections: sections)
        return sections
    }

    /// Sidebar sections in order: "Today", "Yesterday", "Previous 7 Days",
    /// "Previous 30 Days", then one per older month ("September 2026").
    /// Each section keeps the most recently updated first; empty sections
    /// are left out.
    static func grouped(_ conversations: [Conversation], now: Date = Date())
        -> [(title: String, items: [Conversation])] {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: now)

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

    /// Month section titles ("September 2026"). Made once: a formatter is
    /// slow to create.
    private static let monthFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale.autoupdatingCurrent
        formatter.setLocalizedDateFormatFromTemplate("MMMMyyyy")
        return formatter
    }()

    // MARK: - Files

    /// Replaces the list in one change, animated, so the sidebar moves its
    /// rows rather than redrawing them in place, and observers hear about
    /// one change rather than one per step. With Reduce Motion the rows
    /// only fade (`withMotion`).
    private func publish(_ updated: [Conversation]) {
        withMotion(Motion.content) {
            conversations = updated
        }
    }

    private func fileURL(for id: UUID) -> URL? {
        directory?.appendingPathComponent(id.uuidString.lowercased() + ".json")
    }

    private func write(_ conversation: Conversation) {
        guard let url = fileURL(for: conversation.id) else { return }
        Self.onDisk("Save chat") {
            ChatFiles.write(conversation, to: url)
        }
    }

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

/// The file work itself, run on `ChatStore`'s disk queue, off the main
/// thread. The files are the same as when this ran on the main thread:
/// the same encoding, written atomically.
private enum ChatFiles {
    /// Used only on the disk queue, one write at a time.
    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    static func write(_ conversation: Conversation, to url: URL) {
        do {
            let data = try encoder.encode(conversation)
            try data.write(to: url, options: .atomic)
        } catch {
            print("[store] could not save conversation \(conversation.id): \(error.localizedDescription)")
        }
    }

    static func remove(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    static func removeAll(in directory: URL) {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ) else { return }
        for file in files where file.pathExtension == "json" {
            try? FileManager.default.removeItem(at: file)
        }
    }
}
