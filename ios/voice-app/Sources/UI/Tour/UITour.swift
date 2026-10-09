import SwiftUI
import UIKit

/// `-PieUITour 1`: drives the app through its screens on the device and
/// saves a screenshot of each to Documents/tour/NN-name.png, with one JSON
/// line per step in Documents/tour/tour.jsonl, then exits.
///
/// It uses only the controllers and the router, the same entry points the
/// views use, so what it captures is what a person tapping through would
/// see. Every wait has a timeout; a step that times out is captured anyway
/// with a note saying what it was waiting for, so one slow turn never
/// leaves the run hanging.
enum UITour {
    static var isEnabled: Bool {
        UserDefaults.standard.string(forKey: "PieUITour") == "1"
    }

    /// Starts the tour; call after `chat.bootstrap()`.
    @MainActor
    static func run(
        chat: ChatController,
        voice: VoiceModeController,
        router: AppRouter,
        settings: AppSettings,
        store: ChatStore
    ) {
        let script = TourScript(chat: chat, voice: voice, router: router, settings: settings, store: store)
        Task { @MainActor in
            await script.run()
            ChatStore.finishPendingWrites()
            exit(0)
        }
    }
}

@MainActor
private final class TourScript {

    private let chat: ChatController
    private let voice: VoiceModeController
    private let router: AppRouter
    private let settings: AppSettings
    private let store: ChatStore

    private let directory: URL
    private let started = Date()
    private var step = 0
    private var lines: [String] = []

    /// How long a navigation change gets to finish animating before the
    /// capture.
    private let settleTime: TimeInterval = 0.6
    /// A sheet or cover being dismissed has to be gone before the next one
    /// is presented; this is a little longer than the system transition.
    private let dismissTime: TimeInterval = 0.8

    init(chat: ChatController, voice: VoiceModeController, router: AppRouter, settings: AppSettings, store: ChatStore) {
        self.chat = chat
        self.voice = voice
        self.router = router
        self.settings = settings
        self.store = store
        directory = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("tour", isDirectory: true)
    }

    func run() async {
        // A fresh directory per run, so no screenshot from an earlier tour
        // is mistaken for this one's.
        try? FileManager.default.removeItem(at: directory)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        print("[tour] start")

        let booted = await wait(120) { self.chat.engineState == .ready }
        let bootNote = booted
            ? String(format: "engine ready after %.1f s", elapsed)
            : "engine not ready after 120 s: \(chat.engineState)"

        // 01-05: a chat, streamed and finished, in both modes.
        chat.newChat()
        let tourConversation = chat.conversation.id
        await pause(settleTime)
        capture("empty", [bootNote])

        chat.send(text: "What is one good reason to run a language model on a phone?")
        let streamed = await wait(60) { !self.replyText.isEmpty }
        if streamed { await pause(0.5) }
        capture("streaming", streamed
            ? ["phase \(chat.phase)", "\(replyText.count) characters so far"]
            : ["no reply text after 60 s", "phase \(chat.phase)"])

        let answered = await wait(120) { self.chat.phase == .idle }
        await pause(settleTime)
        capture("answered", answered ? [statsNote] : ["reply still running after 120 s", "phase \(chat.phase)"])

        let previousMode = chat.mode
        chat.mode = .thinking
        chat.send(text: "Plan a three-day trip to Pittsburgh on a student budget.")
        let reasoning = await wait(60) { self.isThinkingOrWriting && !self.reasoningText.isEmpty }
        if reasoning { await pause(0.8) }
        capture("thinking", reasoning
            ? ["phase \(chat.phase)", "\(reasoningText.count) characters of reasoning so far"]
            : ["no reasoning after 60 s", "phase \(chat.phase)"])

        let thought = await wait(180) { self.chat.phase == .idle }
        await pause(settleTime)
        capture("thought", thought ? [thoughtNote, statsNote] : ["reply still running after 180 s", "phase \(chat.phase)"])
        chat.mode = previousMode

        // 06-08: the sidebar, the "+" sheet and Settings.
        router.isSidebarOpen = true
        let titled = await wait(8) { !self.tourTitle(tourConversation).isEmpty }
        await pause(settleTime)
        capture("sidebar", titled
            ? ["title \"\(tourTitle(tourConversation))\""]
            : ["no generated title after 8 s"])

        router.isSidebarOpen = false
        await pause(settleTime)
        router.isAttachmentSheetPresented = true
        await pause(dismissTime)
        capture("plus-sheet", [])

        router.isAttachmentSheetPresented = false
        await pause(dismissTime)
        router.isSettingsPresented = true
        await pause(dismissTime)
        capture("settings", [])

        // 09-11: voice mode, listening, answering a sample question, and
        // back to listening.
        router.isSettingsPresented = false
        await pause(dismissTime)
        router.isVoiceModePresented = true
        await pause(1.5)
        capture("voice-listening", ["phase \(voice.phase)", "availability \(String(describing: voice.availability))"])

        if voice.hasSampleQuestions {
            let asked = Date()
            voice.askSampleQuestion()
            var thinkingAt: TimeInterval?
            let speaking = await wait(90) {
                if thinkingAt == nil && self.voice.phase == .thinking {
                    thinkingAt = Date().timeIntervalSince(asked)
                }
                return self.voice.phase == .speaking
            }
            let speakingAt = Date().timeIntervalSince(asked)
            if speaking { await pause(0.6) }
            var notes = speaking
                ? [String(format: "speaking %.1f s after the sample started", speakingAt)]
                : ["not speaking after 90 s", "phase \(voice.phase)"]
            if let thinkingAt {
                notes.append(String(format: "thinking (transcript final) at %.1f s", thinkingAt))
            }
            capture("voice-speaking", notes)
        } else {
            capture("voice-speaking", ["no sample recordings in this build", "phase \(voice.phase)"])
        }

        let listeningAgain = await wait(90) { self.voice.phase == .listening }
        await pause(settleTime)
        // Nobody talks during the tour, so a reply cut short here was cut
        // by its own echo: the false barge-in voice mode must not make.
        let spokenReply = chat.conversation.messages.last { $0.role == .assistant && $0.viaVoice }
        let replyNote = spokenReply.map { "spoken reply stopped early: \($0.wasStopped)" }
            ?? "no spoken reply in the conversation"
        capture("voice-after", listeningAgain
            ? ["user caption \"\(voice.userCaption)\"", "\(voice.assistantCaption.count) characters of reply caption", replyNote]
            : ["not listening again after 90 s", "phase \(voice.phase)", replyNote])
        voice.end()
        router.isVoiceModePresented = false
        await pause(dismissTime)

        // 12-14: the chat after voice mode, dark appearance, temporary chat.
        capture("chat-after-voice", ["\(chat.conversation.messages.count) messages in the conversation"])

        let previousAppearance = settings.appearance
        settings.appearance = .dark
        await pause(settleTime)
        capture("dark", ["appearance restored to \(previousAppearance.rawValue) after the capture"])
        settings.appearance = previousAppearance
        await pause(settleTime)

        chat.newChat(temporary: true)
        await pause(settleTime)
        capture("temporary", ["temporary \(chat.conversation.isTemporary)"])

        // The tour's chat was only ever scaffolding for the screenshots;
        // repeated runs should not fill the sidebar with copies of it.
        store.delete(tourConversation)
        print(String(format: "[tour] finished: %d steps in %.1f s", step, elapsed))
    }

    // MARK: - Reading the chat

    private var lastAssistant: StoredMessage? {
        guard let last = chat.conversation.messages.last, last.role == .assistant else { return nil }
        return last
    }

    private var replyText: String { lastAssistant?.text ?? "" }

    private var reasoningText: String { lastAssistant?.reasoning ?? "" }

    private var isThinkingOrWriting: Bool {
        switch chat.phase {
        case .thinking, .writing: return true
        case .idle, .waiting: return false
        }
    }

    private func tourTitle(_ id: UUID) -> String {
        if chat.conversation.id == id, !chat.conversation.title.isEmpty {
            return chat.conversation.title
        }
        return store.conversation(id)?.title ?? ""
    }

    private var statsNote: String {
        guard let stats = lastAssistant?.stats else { return "no stats on the reply" }
        return String(
            format: "generated %d, reused %d of %d prompt tokens, ttft %.2f s, %.1f tok/s",
            stats.generated,
            stats.reused,
            stats.promptTokens,
            stats.timeToFirstToken ?? 0,
            stats.tokensPerSecond
        )
    }

    private var thoughtNote: String {
        guard let seconds = lastAssistant?.thoughtSeconds else { return "no thought duration" }
        return String(format: "thought for %.1f s", seconds)
    }

    // MARK: - Waiting

    private var elapsed: TimeInterval { Date().timeIntervalSince(started) }

    /// Polls `condition` every 100 ms until it holds or `timeout` passes.
    /// Returns whether it held.
    private func wait(_ timeout: TimeInterval, until condition: () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() >= deadline { return false }
            await pause(0.1)
        }
        return true
    }

    private func pause(_ seconds: TimeInterval) async {
        try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
    }

    // MARK: - Capturing

    private struct Line: Encodable {
        let step: Int
        let name: String
        let t: Double
        let notes: String
    }

    /// Saves the key window, which holds any sheet or cover presented over
    /// the chat, as the next step's PNG and appends its line to the log.
    private func capture(_ name: String, _ notes: [String]) {
        step += 1
        let stem = String(format: "%02d-%@", step, name)
        var notes = notes

        if let window = keyWindow {
            let renderer = UIGraphicsImageRenderer(bounds: window.bounds)
            let image = renderer.image { _ in
                _ = window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
            }
            if let png = image.pngData() {
                do {
                    try png.write(to: directory.appendingPathComponent("\(stem).png"))
                } catch {
                    notes.append("screenshot not written: \(error.localizedDescription)")
                }
            } else {
                notes.append("screenshot could not be encoded")
            }
        } else {
            notes.append("no key window to capture")
        }

        let line = Line(
            step: step,
            name: stem,
            t: (elapsed * 10).rounded() / 10,
            notes: notes.joined(separator: "; ")
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        if let data = try? encoder.encode(line), let json = String(data: data, encoding: .utf8) {
            lines.append(json)
            // The whole log each time, so whatever ran before a crash is
            // already on disk.
            try? (lines.joined(separator: "\n") + "\n")
                .write(to: directory.appendingPathComponent("tour.jsonl"), atomically: true, encoding: .utf8)
        }
        print("[tour] \(stem): \(line.notes)")
    }

    private var keyWindow: UIWindow? {
        UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .flatMap(\.windows)
            .first(where: \.isKeyWindow)
    }
}
