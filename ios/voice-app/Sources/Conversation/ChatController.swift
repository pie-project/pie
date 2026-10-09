import Combine
import Foundation
import SwiftUI

/// The open conversation: sending, streaming, stopping, regenerating,
/// editing, attachments, read-aloud, and the engine's boot state.
///
/// Holds the model as a `ConversationBackend` and the speaker as a
/// `SpeechOutput`; it takes prompts and generation presets from
/// `PieRuntimeConfig` but knows nothing else about the engine.
///
/// Every reply carries a `ReplyTicket`, and every callback a reply makes
/// is fenced by it: a reply that was stopped, or whose conversation was
/// left, can never write into the conversation on screen afterwards.
///
/// Motion: the moments a person notices that change a row in place (the
/// first words of a reply, a reply finishing or stopped, a thumbs tap,
/// read-aloud starting) are made inside `withMotion`, so the views'
/// transitions run. Changes that add or remove rows (a send, a regenerate,
/// an edit) are not: the transcript glides the question to the top as they
/// happen, and a scroll requested while its content lays out inside an
/// animation is dropped (measured on iOS 26), so those rows animate
/// themselves instead (see `MessageList`). Streamed text never animates
/// through a transaction: a chat reply's text goes through a `RevealPacer`,
/// which shows it a word at a time at most once per frame, and the
/// transcript fades the new words in itself.
@MainActor
final class ChatController: ObservableObject {

    enum EngineState: Equatable {
        case booting(seconds: Int)
        case ready
        case failed(String)
    }

    enum ReplyPhase: Equatable {
        case idle
        /// Sent; nothing has streamed back yet.
        case waiting
        /// Streaming reasoning (thinking mode).
        case thinking(since: Date)
        /// Streaming the reply.
        case writing
    }

    @Published private(set) var conversation = Conversation()
    @Published private(set) var engineState: EngineState = .booting(seconds: 0)
    @Published private(set) var phase: ReplyPhase = .idle
    /// The composer's text.
    @Published var draft = ""
    @Published private(set) var pendingAttachments: [Attachment] = []
    /// An attachment is being read (OCR or document text).
    @Published private(set) var isImportingAttachment = false
    @Published var mode: ReplyMode = .instant
    /// The message being read aloud, if any.
    @Published private(set) var readingAloudMessageID: UUID?
    /// A transient notice for the top of the chat ("Couldn't read that
    /// file"); the UI shows it and sets it back to nil.
    @Published var banner: String?

    var isGenerating: Bool { phase != .idle }

    var canSend: Bool {
        engineState == .ready
            && !isGenerating
            && !isImportingAttachment
            && (!draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !pendingAttachments.isEmpty)
    }

    /// "Qwen3.5-0.8B · Metal · inferlets on wasmtime Pulley".
    var engineDescription: String { backend.engineDescription }

    private let store: ChatStore
    private let backend: ConversationBackend
    private let speech: SpeechOutput
    private let settings: AppSettings

    /// The reply being generated, if any. Its ticket is the fence.
    private var active: ActiveReply?
    private var didStartBoot = false
    /// The open conversation has been written to the store, so its
    /// disappearing from the store means the user deleted it.
    private var isSaved = false
    private var importsInFlight = 0
    /// Bumped whenever the composer is emptied for another conversation,
    /// so an attachment still being read lands only in the composer it
    /// was added from.
    private var composerGeneration: UInt64 = 0
    private var titlesInFlight: Set<UUID> = []
    private var storeObservation: AnyCancellable?
    /// The composer's dictation is open; see `DictationController.activityDidChange`.
    private var isDictationOpen = false
    private var dictationObservation: NSObjectProtocol?

    /// Attachments one message can carry. They share one slice of the
    /// context (`PieRuntimeConfig.attachmentTokensPerMessage`), so past a
    /// handful each would be cut to a few lines.
    private static let attachmentLimit = 5

    init(store: ChatStore, backend: ConversationBackend, speech: SpeechOutput, settings: AppSettings) {
        self.store = store
        self.backend = backend
        self.speech = speech
        self.settings = settings
        mode = settings.defaultMode

        // Renames and deletes from the sidebar go straight to the store;
        // the open conversation follows them. Delivered on the next turn
        // of the main queue so the store's array has been updated by the
        // time it is read.
        storeObservation = store.$conversations
            .dropFirst()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                MainActor.assumeIsolated { self?.followStore() }
            }

        // Posted on the main thread, so delivered synchronously there.
        dictationObservation = NotificationCenter.default.addObserver(
            forName: DictationController.activityDidChange,
            object: nil,
            queue: nil
        ) { [weak self] note in
            MainActor.assumeIsolated {
                let dictating = (note.object as? DictationController)?.isDictating ?? false
                self?.dictationChanged(dictating)
            }
        }
    }

    deinit {
        if let dictationObservation {
            NotificationCenter.default.removeObserver(dictationObservation)
        }
    }

    // MARK: - Engine

    /// Boots the engine (warm-up turn) and ticks `engineState` while it
    /// does, so a hang is visibly different from a slow load.
    func bootstrap() {
        guard !didStartBoot else { return }
        didStartBoot = true
        engineState = .booting(seconds: 0)

        let started = Date()
        let ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard let self, !Task.isCancelled, case .booting = self.engineState else { return }
                self.engineState = .booting(seconds: Int(-started.timeIntervalSinceNow))
            }
        }
        Task {
            let failure = await backend.warmUp()
            ticker.cancel()
            engineState = failure.map(EngineState.failed) ?? .ready
        }
    }

    /// A failed boot leaves half-initialised engine state behind that the
    /// shim refuses to boot over in the same process, so recovery is a
    /// clean relaunch: quit now, and the next tap on the icon starts from
    /// scratch.
    func quitToRetryEngineBoot() {
        guard case .failed(let reason) = engineState else { return }
        print("[app] quitting for a fresh engine boot after: \(reason)")
        fflush(stdout)
        ChatStore.finishPendingWrites()
        exit(0)
    }

    // MARK: - Conversations

    func newChat(temporary: Bool = false) {
        leaveConversation()
        conversation = Conversation(isTemporary: temporary)
        isSaved = false
        mode = settings.defaultMode
    }

    func open(_ id: UUID) {
        guard id != conversation.id, let saved = store.conversation(id) else { return }
        leaveConversation()
        conversation = saved
        isSaved = true
        mode = settings.defaultMode
    }

    /// Stops whatever the open conversation is doing and saves it as it
    /// stands, a stopped reply's partial text included.
    private func leaveConversation() {
        stopReadingAloud()
        stop()
        draft = ""
        pendingAttachments = []
        composerGeneration += 1
        save()
    }

    private func followStore() {
        guard isSaved else { return }
        guard let saved = store.conversation(conversation.id) else {
            // Deleted while open: whatever it was doing has nowhere to go.
            if let reply = active {
                backend.cancel(reply.ticket)
                reply.stopPacing()
                active = nil
                phase = .idle
            }
            stopReadingAloud()
            conversation = Conversation()
            isSaved = false
            mode = settings.defaultMode
            draft = ""
            pendingAttachments = []
            composerGeneration += 1
            return
        }
        if !saved.title.isEmpty, saved.title != conversation.title {
            conversation.title = saved.title
        }
    }

    private func save() {
        guard !conversation.isTemporary, !conversation.isEmpty else { return }
        // Saved before and gone from the store now: deleted from the
        // sidebar or Settings, which then move on with `newChat()`. Saving
        // here would bring it back.
        if isSaved, store.conversation(conversation.id) == nil { return }
        store.save(conversation)
        isSaved = true
    }

    // MARK: - Sending

    /// Sends `draft` and `pendingAttachments` as a user message.
    func send() {
        guard canSend else { return }
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachments = pendingAttachments
        draft = ""
        pendingAttachments = []
        submit(text, attachments: attachments)
    }

    /// Sends `text` as a user message (suggestion chips, the tour).
    func send(text: String) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, engineState == .ready, !isGenerating else { return }
        submit(text, attachments: [])
    }

    /// Not animated here: the rows animate themselves in (the bubble rises,
    /// the pending dot appears just after it) while the list glides the
    /// question to the top.
    private func submit(_ text: String, attachments: [Attachment]) {
        conversation.messages.append(StoredMessage(role: .user, text: text, attachments: attachments))
        conversation.updatedAt = Date()
        startChatReply(mode: mode)
    }

    /// Stops the reply being generated; its partial text stays.
    ///
    /// The message is settled here and now rather than when the engine
    /// confirms the cancel: the user sees it stop at once, and a question
    /// asked straight after (voice mode's barge-in) is accepted at once and
    /// simply queues behind the cancelled reply's last ~200 ms.
    ///
    /// A chat reply keeps exactly the text on screen: words the pacer had
    /// not shown yet are dropped, nothing shown is taken back. The dot or
    /// shimmer fades out and the reply's controls fade in.
    func stop() {
        stop(animated: true)
    }

    /// `stop()`, optionally without its animation: an edit stops the reply
    /// it is about to replace, and the transcript's glide to the edited
    /// message must not run into an animated layout (see `MessageList`).
    private func stop(animated: Bool) {
        guard let reply = active else { return }
        backend.cancel(reply.ticket)
        reply.stopPacing()
        active = nil
        guard let index = index(of: reply.messageID) else {
            phase = .idle
            return
        }
        let settle = {
            self.phase = .idle
            self.conversation.messages[index].isStreaming = false
            self.conversation.messages[index].wasStopped = true
            self.conversation.messages[index].thoughtSeconds = reply.thoughtSeconds(endingAt: Date())
            self.conversation.updatedAt = Date()
            reply.finished = self.conversation.messages[index]
        }
        if animated {
            withMotion(Motion.fadeIn, settle)
        } else {
            settle()
        }
        save()
    }

    /// Replaces an assistant reply with a fresh one, optionally in a
    /// different mode.
    func regenerate(_ messageID: UUID, mode: ReplyMode? = nil) {
        guard engineState == .ready, !isGenerating,
              let index = conversation.messages.firstIndex(where: { $0.id == messageID && $0.role == .assistant }),
              let question = conversation.messages[..<index].lastIndex(where: { $0.role == .user })
        else { return }
        // Not animated here: the old reply fades out where it is on its
        // own transition while the new one's pending dot pops in in its
        // place, and the list brings the question to the top.
        conversation.messages.removeSubrange((question + 1)...)
        conversation.updatedAt = Date()
        startChatReply(mode: mode ?? self.mode)
    }

    /// Rewrites a user message and answers again from there; everything
    /// after it is dropped. As in ChatGPT, an edit is answered at once: a
    /// reply still on its way is stopped first (the edit drops it anyway).
    func edit(_ messageID: UUID, to newText: String) {
        stop(animated: false)
        let text = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard engineState == .ready, !isGenerating,
              let index = conversation.messages.firstIndex(where: { $0.id == messageID && $0.role == .user }),
              !text.isEmpty || !conversation.messages[index].attachments.isEmpty
        else { return }
        // Behind the closing sheet, and not animated here: the turns after
        // the message fade out on their own transition, a fresh reply
        // starts below it, and the list glides the edited message to the
        // top.
        conversation.messages[index].text = text
        conversation.messages.removeSubrange((index + 1)...)
        conversation.updatedAt = Date()
        startChatReply(mode: mode)
    }

    /// The chosen thumb fills and the other one fades away.
    func setFeedback(_ feedback: Feedback?, for messageID: UUID) {
        guard let index = index(of: messageID) else { return }
        withMotion(Motion.control) {
            conversation.messages[index].feedback = feedback
        }
        save()
    }

    /// Voice mode's turn: sends `text` as a user message in this
    /// conversation with the spoken-reply prompt and options, streams reply
    /// text to `onText` (on the main queue), and returns the finished
    /// assistant message, or nil if the turn failed. A `stop()` during the
    /// turn returns the partial message with `wasStopped == true`.
    func sendSpoken(_ text: String, onText: @escaping (String) -> Void) async -> StoredMessage? {
        let utterance = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !utterance.isEmpty, engineState == .ready else { return nil }
        // A spoken question supersedes a typed reply still generating.
        stop()
        conversation.messages.append(StoredMessage(role: .user, text: utterance, viaVoice: true))
        conversation.updatedAt = Date()
        let system = PieRuntimeConfig.systemPrompt(
            PieRuntimeConfig.chatSystemPrompt,
            customInstructions: settings.customInstructions
        )
        return await startReply(
            options: PieRuntimeConfig.voiceOptions,
            systemPrompt: system,
            viaVoice: true,
            onText: onText
        ).value
    }

    /// Takes back voice mode's last exchange, the spoken `question` and
    /// the reply nobody heard, when the user turns out to have paused in
    /// the middle of the question: voice mode asks it again together with
    /// what they said next, and the transcript should show the one
    /// question. Does nothing unless the conversation ends with exactly
    /// that exchange and its reply is no longer generating.
    func withdrawSpokenExchange(asking question: String) {
        let messages = conversation.messages
        guard active == nil, messages.count >= 2 else { return }
        let asked = messages[messages.count - 2]
        let reply = messages[messages.count - 1]
        guard asked.role == .user, asked.viaVoice, asked.text == question,
              reply.role == .assistant, reply.viaVoice
        else { return }
        conversation.messages.removeLast(2)
        conversation.updatedAt = Date()
        if conversation.isEmpty {
            // Saved with this exchange alone; an empty conversation is
            // never saved, so the old copy is removed instead. Not saved
            // any more, so the open conversation does not treat its
            // leaving the store as a delete from the sidebar.
            if isSaved {
                isSaved = false
                store.delete(conversation.id)
            }
        } else {
            save()
        }
    }

    // MARK: - Generation

    private func startChatReply(mode: ReplyMode) {
        let system = PieRuntimeConfig.systemPrompt(
            PieRuntimeConfig.chatSystemPrompt,
            customInstructions: settings.customInstructions
        )
        startReply(
            options: PieRuntimeConfig.chatOptions(for: mode),
            systemPrompt: system,
            viaVoice: false,
            onText: nil
        )
    }

    /// Answers the user message at the end of the conversation.
    ///
    /// Everything up to the backend call happens before this returns, so
    /// `isGenerating` is already true for the caller and a second tap on
    /// send cannot start a second reply.
    @discardableResult
    private func startReply(
        options: ReplyOptions,
        systemPrompt: String,
        viaVoice: Bool,
        onText: ((String) -> Void)?
    ) -> Task<StoredMessage?, Never> {
        stopReadingAloud()
        let prompt = MessageRendering.prompt(system: systemPrompt, messages: conversation.messages)
        let placeholder = StoredMessage(role: .assistant, text: "", viaVoice: viaVoice, isStreaming: true)
        conversation.messages.append(placeholder)
        conversation.updatedAt = Date()
        phase = .waiting
        // Saved with the reply row in place, so a question whose answer is
        // cut short by the app being killed comes back with a stopped
        // reply that Regenerate can replace, not as a question nobody can
        // answer again.
        save()

        let reply = ActiveReply(messageID: placeholder.id, viaVoice: viaVoice, onText: onText)
        if !viaVoice {
            reply.textPacer = RevealPacer { [weak self, weak reply] shown in
                guard let self, let reply else { return }
                self.reveal(text: shown, of: reply)
            }
            reply.reasoningPacer = RevealPacer { [weak self, weak reply] shown in
                guard let self, let reply else { return }
                self.reveal(reasoning: shown, of: reply)
            }
            Haptics.prepareStreaming(enabled: settings.haptics)
        }
        active = reply
        let session = conversation.sessionKey
        return Task { await stream(reply, session: session, prompt: prompt, options: options) }
    }

    private func stream(
        _ reply: ActiveReply,
        session: String,
        prompt: [PromptMessage],
        options: ReplyOptions
    ) async -> StoredMessage? {
        // The backend calls back on its own thread; the stream hands the
        // events to the main actor in order.
        let (events, sink) = AsyncStream.makeStream(of: ReplyEvent.self)
        let backend = self.backend
        let ticket = reply.ticket
        let generation = Task { () -> Result<ReplyResult, Error> in
            defer { sink.finish() }
            do {
                return .success(try await backend.reply(
                    ticket,
                    session: session,
                    messages: prompt,
                    options: options,
                    onEvent: { sink.yield($0) }
                ))
            } catch {
                return .failure(error)
            }
        }

        for await event in events where active === reply {
            apply(event, to: reply)
        }
        let outcome = await generation.value

        // Stopped, or its conversation was left: already settled, or
        // nothing left to settle.
        guard active === reply else { return reply.finished }

        switch outcome {
        case .success(let result):
            // The screen may still be a few words behind the engine: they
            // are shown within a fraction of a second and finish fading in
            // before the reply settles, so its controls arrive after the
            // last word. A stop meanwhile settles it instead.
            await catchUp(reply, with: result)
            guard active === reply else { return reply.finished }
            reply.stopPacing()
            active = nil
            withMotion(Motion.fadeIn) {
                phase = .idle
                finish(reply, with: result)
            }
            if !result.cancelled { nameConversationIfNeeded() }
            return reply.finished
        case .failure(let error):
            // Keep everything that arrived before the failure.
            reply.reasoningPacer?.flush()
            reply.textPacer?.flush()
            reply.stopPacing()
            active = nil
            fail(reply, error: error)
            return nil
        }
    }

    /// Shows what the screen has not caught up with yet: the end of the
    /// reasoning at once (it is folded away by now), the end of the reply
    /// at the quick finishing pace, then waits out the last words' fade.
    private func catchUp(_ reply: ActiveReply, with result: ReplyResult) async {
        guard let textPacer = reply.textPacer else { return }
        if let reasoningPacer = reply.reasoningPacer {
            if !result.reasoning.isEmpty, !result.cancelled {
                reasoningPacer.catchUp(to: String(result.reasoning.drop(while: \.isWhitespace)))
            }
            reasoningPacer.flush()
        }
        if !result.cancelled {
            textPacer.catchUp(to: String(result.text.drop(while: \.isWhitespace)))
        }
        guard textPacer.hasReceived else { return }
        await textPacer.finish()
        let fadeLeft = Self.lastWordsFade - textPacer.timeSinceLastReveal
        guard active === reply, fadeLeft > 0 else { return }
        try? await Task.sleep(nanoseconds: UInt64(fadeLeft * 1_000_000_000))
    }

    /// How long the last revealed words take to fade in fully; matches
    /// `RevealText.fadeDuration`.
    private static let lastWordsFade: TimeInterval = 0.25

    /// One streamed event. A chat reply's text goes to its pacer, which
    /// puts it on screen through `reveal(text:of:)`; voice mode's goes into
    /// the message straight away, as it always has.
    private func apply(_ event: ReplyEvent, to reply: ActiveReply) {
        guard let index = index(of: reply.messageID) else { return }
        switch event {
        case .reasoning(var chunk):
            let isFirst = reply.reasoningPacer.map { !$0.hasReceived }
                ?? conversation.messages[index].reasoning.isEmpty
            if isFirst {
                chunk = String(chunk.drop(while: \.isWhitespace))
                guard !chunk.isEmpty else { return }
            }
            if reply.firstReasoningAt == nil {
                let now = Date()
                reply.firstReasoningAt = now
                if reply.reasoningPacer == nil { phase = .thinking(since: now) }
            }
            if let pacer = reply.reasoningPacer {
                pacer.receive(chunk)
            } else {
                conversation.messages[index].reasoning += chunk
            }

        case .text(var chunk):
            // Leading whitespace would show as a blank line at the top of
            // the bubble until the authoritative text replaced it.
            let isFirst = reply.textPacer.map { !$0.hasReceived }
                ?? conversation.messages[index].text.isEmpty
            if isFirst {
                chunk = String(chunk.drop(while: \.isWhitespace))
                guard !chunk.isEmpty else { return }
            }
            if reply.firstTextAt == nil {
                let now = Date()
                reply.firstTextAt = now
                if reply.textPacer == nil {
                    conversation.messages[index].thoughtSeconds = reply.thoughtSeconds(endingAt: now)
                    phase = .writing
                }
                // Thinking is over; the rest of it is folded away anyway.
                reply.reasoningPacer?.flush()
            }
            if let pacer = reply.textPacer {
                pacer.receive(chunk)
            } else {
                conversation.messages[index].text += chunk
            }
            reply.onText?(chunk)
        }
    }

    /// The pacer has more of a chat reply's text for the screen.
    ///
    /// The first words are a moment of their own: the pending dot (or the
    /// "Thinking" shimmer) gives way to them and the phase becomes
    /// `.writing`, animated. After that the text is set plainly; the
    /// transcript fades each new word in itself. Each reveal is one tick
    /// of the soft haptic train.
    private func reveal(text shown: String, of reply: ActiveReply) {
        guard active === reply, let index = index(of: reply.messageID) else { return }
        if conversation.messages[index].text.isEmpty {
            let wasThinking = phase != .waiting
            withMotion(wasThinking ? Motion.crossfade : Motion.fadeOut) {
                conversation.messages[index].thoughtSeconds = reply.thoughtSeconds(endingAt: reply.firstTextAt ?? Date())
                conversation.messages[index].text = shown
                phase = .writing
            }
        } else {
            conversation.messages[index].text = shown
        }
        Haptics.streamTick(enabled: settings.haptics)
    }

    /// The pacer has more of a chat reply's reasoning. The first of it
    /// turns the pending dot into the "Thinking" shimmer.
    private func reveal(reasoning shown: String, of reply: ActiveReply) {
        guard active === reply, let index = index(of: reply.messageID) else { return }
        if conversation.messages[index].reasoning.isEmpty, phase == .waiting {
            withMotion(Motion.crossfade) {
                phase = .thinking(since: reply.firstReasoningAt ?? Date())
                conversation.messages[index].reasoning = shown
            }
        } else {
            conversation.messages[index].reasoning = shown
        }
    }

    private func finish(_ reply: ActiveReply, with result: ReplyResult) {
        guard let index = index(of: reply.messageID) else { return }
        var message = conversation.messages[index]
        // The inferlet's return value is the authoritative text; the
        // streamed text can lag it by a token.
        if !result.cancelled, !result.text.isEmpty {
            message.text = result.text
        }
        if !result.reasoning.isEmpty {
            message.reasoning = result.reasoning
        }
        message.stats = result.stats
        message.thoughtSeconds = reply.thoughtSeconds(endingAt: Date())
        message.wasStopped = result.cancelled
        message.isStreaming = false
        // A turn that generated nothing is a real failure mode worth
        // seeing, not a blank bubble.
        if !result.cancelled, message.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            message.text = MessageRendering.noReplyPlaceholder
        }
        conversation.messages[index] = message
        conversation.updatedAt = Date()
        reply.finished = message
        save()
    }

    /// The engine could not produce the reply. The row stays, so the
    /// question still has an answer slot with Regenerate under it: what
    /// streamed before the failure is kept and marked cut off, and an empty
    /// row says something went wrong. The engine's own message is for the
    /// log; the banner says what the user can do.
    private func fail(_ reply: ActiveReply, error: Error) {
        print("[chat] reply failed: \(error.localizedDescription)")
        banner = reply.viaVoice
            ? "Pie couldn't answer that. Try asking again."
            : "Pie couldn't finish that reply. Tap Regenerate to try again."
        guard let index = index(of: reply.messageID) else {
            phase = .idle
            return
        }
        var message = conversation.messages[index]
        message.isStreaming = false
        message.thoughtSeconds = reply.thoughtSeconds(endingAt: Date())
        if message.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            message.text = MessageRendering.failedReplyPlaceholder
        } else {
            message.wasStopped = true
        }
        withMotion(Motion.fadeIn) {
            phase = .idle
            conversation.messages[index] = message
            conversation.updatedAt = Date()
        }
        save()
    }

    private func index(of messageID: UUID) -> Int? {
        conversation.messages.firstIndex { $0.id == messageID }
    }

    // MARK: - Titles

    /// Names a saved conversation after its first completed exchange.
    ///
    /// A separate reply on its own session, launched only once the
    /// exchange's reply has returned, so it never delays a reply the user
    /// is waiting for; a question sent meanwhile queues behind its few
    /// tokens.
    private func nameConversationIfNeeded() {
        let id = conversation.id
        guard !conversation.isTemporary,
              conversation.title.isEmpty,
              !titlesInFlight.contains(id),
              let first = conversation.messages.first(where: { $0.role == .user })
        else { return }
        titlesInFlight.insert(id)

        let opening = first.text.isEmpty ? MessageRendering.content(of: first) : first.text
        let request = "Write a title of at most five words for a conversation that starts with: "
            + String(opening.prefix(300))
            + ". Reply with the title only."
        let messages = [
            PromptMessage(role: .system, content: PieRuntimeConfig.titleSystemPrompt),
            PromptMessage(role: .user, content: request),
        ]
        let backend = self.backend

        Task {
            let generated = try? await backend.reply(
                ReplyTicket(),
                session: PieRuntimeConfig.titleSessionName,
                messages: messages,
                options: PieRuntimeConfig.titleOptions,
                onEvent: { _ in }
            )
            titlesInFlight.remove(id)
            let title = generated.flatMap { MessageRendering.cleanTitle($0.text) }
            applyTitle(title, to: id)
        }
    }

    private func applyTitle(_ title: String?, to id: UUID) {
        if conversation.id == id {
            guard conversation.title.isEmpty, !conversation.isEmpty else { return }
            conversation.title = title ?? conversation.displayTitle
            save()
        } else if let saved = store.conversation(id), saved.title.isEmpty {
            store.rename(id, to: title ?? saved.displayTitle)
        }
    }

    // MARK: - Read aloud

    /// Starts reading a message aloud, or stops it if it is the one
    /// being read.
    func toggleReadAloud(_ messageID: UUID) {
        if readingAloudMessageID == messageID {
            stopReadingAloud()
            return
        }
        guard let index = index(of: messageID), conversation.messages[index].role == .assistant else { return }
        let spoken = MessageRendering.speakable(conversation.messages[index].text)
        guard !spoken.isEmpty, !MessageRendering.isPlaceholder(spoken) else { return }
        guard !isDictationOpen else {
            banner = "Finish dictating to hear this read aloud"
            return
        }

        stopReadingAloud()
        speech.delegate = self
        speech.voiceIdentifier = settings.voiceIdentifier
        speech.rate = settings.speechRate
        withMotion(Motion.control) { readingAloudMessageID = messageID }

        let chunker = SentenceChunker()
        for sentence in chunker.push(spoken) {
            speech.enqueue(sentence)
        }
        if let rest = chunker.flush() {
            speech.enqueue(rest)
        }
        speech.finishTurn()
    }

    /// Silences read-aloud, if it is playing. Voice mode calls this before
    /// it takes the speaker over.
    func stopReadingAloud() {
        guard readingAloudMessageID != nil else { return }
        withMotion(Motion.control) { readingAloudMessageID = nil }
        speech.stop()
    }

    /// Dictation listens on the raw microphone, so read-aloud stops when
    /// it opens and is refused until it closes; otherwise the speaker's
    /// words would be typed into the draft.
    private func dictationChanged(_ open: Bool) {
        isDictationOpen = open
        if open { stopReadingAloud() }
    }

    // MARK: - Attachments

    func addPhoto(_ imageData: Data) async {
        await importAttachment { try await AttachmentImporter.photo(imageData) }
    }

    func addFile(at url: URL) async {
        await importAttachment { try await AttachmentImporter.file(at: url) }
    }

    func removePendingAttachment(_ id: UUID) {
        pendingAttachments.removeAll { $0.id == id }
    }

    private func importAttachment(_ load: () async throws -> Attachment) async {
        guard pendingAttachments.count + importsInFlight < Self.attachmentLimit else {
            banner = "Up to \(Self.attachmentLimit) attachments per message"
            return
        }
        let generation = composerGeneration
        importsInFlight += 1
        isImportingAttachment = true
        defer {
            importsInFlight -= 1
            isImportingAttachment = importsInFlight > 0
        }
        do {
            let attachment = try await load()
            // Read for a conversation the user has since left.
            guard composerGeneration == generation else { return }
            pendingAttachments.append(attachment)
        } catch {
            guard composerGeneration == generation else { return }
            banner = error.localizedDescription
        }
    }

    // MARK: - Sharing

    /// The conversation as Markdown, for the share sheet.
    func shareText() -> String {
        MessageRendering.shareText(conversation)
    }
}

// MARK: - SpeechOutputDelegate (read-aloud)

extension ChatController: @preconcurrency SpeechOutputDelegate {
    func speechOutputDidStart() {}

    func speechOutputDidFinish(interrupted: Bool) {
        // A finish that arrives after a different message has started
        // (the old one being cut off) must not clear the new one.
        guard !speech.isSpeaking else { return }
        withMotion(Motion.control) { readingAloudMessageID = nil }
    }

    func speechOutputLevel(_ level: Float) {}
}

/// One reply in flight: its ticket, the message it streams into, and the
/// timings the message's "Thought for Ns" is computed from.
private final class ActiveReply {
    let ticket = ReplyTicket()
    let messageID: UUID
    /// Voice mode's turn, which says why it failed in its own words.
    let viaVoice: Bool
    let onText: ((String) -> Void)?
    var firstReasoningAt: Date?
    var firstTextAt: Date?
    /// The message as it was settled; what a caller awaiting the reply
    /// gets back.
    var finished: StoredMessage?
    /// Meter a chat reply's text and reasoning onto the screen. Nil for
    /// voice mode's turns, which are written as they arrive.
    var textPacer: RevealPacer?
    var reasoningPacer: RevealPacer?

    init(messageID: UUID, viaVoice: Bool, onText: ((String) -> Void)?) {
        self.messageID = messageID
        self.viaVoice = viaVoice
        self.onText = onText
    }

    /// From the first reasoning to the first reply text, or to `end` if no
    /// reply text came. Nil when the model did not reason.
    func thoughtSeconds(endingAt end: Date) -> Double? {
        guard let start = firstReasoningAt else { return nil }
        return (firstTextAt ?? end).timeIntervalSince(start)
    }

    /// No more reveals: the screen keeps what it shows.
    @MainActor
    func stopPacing() {
        textPacer?.stop()
        reasoningPacer?.stop()
    }
}
