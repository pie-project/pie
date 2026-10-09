import Combine
import Foundation

/// `-PieVoiceSoak 1`: asks voice mode the bundled sample questions round
/// after round, with nobody near the phone, and reports every turn voice
/// mode took that nobody asked for.
///
/// The failure it hunts is a reply answering itself: its echo leaks past
/// the echo canceller, the recogniser makes words of it, and voice mode
/// takes them for the user, cutting the reply off as a barge-in or asking
/// them as the next question. With nobody talking, the conversation must
/// end up holding exactly one spoken question per round, each a sample's
/// transcript, and every reply in full; anything else came out of the
/// speaker. `-PieVoiceSoakRounds n` sets the number of rounds; the default
/// of six asks each of the three recordings twice.
///
/// A clean conversation does not on its own show that voice mode can tell
/// its echo from the user: a run in which no echo reached the recogniser
/// is just as clean. So each round also lists what the microphone told
/// voice mode, utterance by utterance, with what voice mode took from each
/// as the user's words, and every spell of the reply being ducked because
/// the microphone heard something. With nobody near the phone, whatever
/// the microphone makes words of outside the question is the reply heard
/// back, or the room. The summary's `result` says which kind of pass a
/// clean run was. Ducking does not fail a run, since a moment of quieter
/// speech is what voice mode accepts instead of cutting a reply off on a
/// noise, but a run that ducks every reply is one to look into.
///
/// It drives voice mode only through the router and the controller's
/// intents, as the screens do. It watches the controller's published phase
/// and caption, the synthesizer's ducking, and the microphone's calls to
/// voice mode through a tap that passes each one straight on, so what it
/// measures is the app as a person uses it. One JSON line per round and a
/// summary go to Documents/voice-soak.jsonl and to stdout as
/// `PIESOAK {json}`; times in a round are seconds from its sample being
/// asked for. Every wait has a limit and leaves a note when it runs out,
/// so the run always ends. The soak's conversation is deleted afterwards,
/// so repeated runs do not fill the sidebar.
enum VoiceSoak {
    static var isEnabled: Bool {
        UserDefaults.standard.string(forKey: "PieVoiceSoak") == "1"
    }

    private static var roundCount: Int {
        let n = UserDefaults.standard.integer(forKey: "PieVoiceSoakRounds")
        return n > 0 ? n : 6
    }

    /// Starts the soak; call after `chat.bootstrap()`.
    @MainActor
    static func run(
        chat: ChatController,
        voice: VoiceModeController,
        speech: SpeechOutput,
        microphone: SpeechInput,
        router: AppRouter,
        store: ChatStore
    ) {
        let script = SoakScript(
            chat: chat,
            voice: voice,
            speech: speech,
            microphone: microphone,
            router: router,
            store: store,
            rounds: roundCount
        )
        Task { @MainActor in
            await script.run()
            ChatStore.finishPendingWrites()
            exit(0)
        }
    }
}

@MainActor
private final class SoakScript {

    /// A cold launch pages the whole model in before the warm-up turn.
    private static let bootTimeout: TimeInterval = 180
    /// For the cover to appear and start voice mode.
    private static let presentTimeout: TimeInterval = 10
    /// For the permission check and the echo-cancelled engine to come up.
    private static let microphoneTimeout: TimeInterval = 20
    /// Longer than the detector takes to measure the room, so the first
    /// recording plays against a session that has settled, as later ones do.
    private static let settleTime: TimeInterval = 1.0
    /// A recording lasts a few seconds, and its transcript is final at
    /// most two seconds after it ends.
    private static let questionTimeout: TimeInterval = 30
    /// From the question to the reply's last word.
    private static let replyTimeout: TimeInterval = 60
    /// The quiet after a reply in which a turn nobody asked for would show
    /// itself: an utterance made of the reply's tail ends 1.1 s after the
    /// sound does, and its transcript follows within a second or two.
    private static let quietPeriod: TimeInterval = 4
    /// For voice mode to be quiet that long, before giving up on it.
    private static let quietTimeout: TimeInterval = 60
    /// A cover being dismissed has to be gone before the chat is changed
    /// under it.
    private static let dismissTime: TimeInterval = 0.8

    private let chat: ChatController
    private let voice: VoiceModeController
    private let speech: SpeechOutput
    private let microphone: SpeechInput
    private let router: AppRouter
    private let store: ChatStore
    private let rounds: Int

    private let log = SoakLog()
    private let started = Date()
    private let tap = MicrophoneTap()

    /// Every change of voice mode's phase, oldest first, with the user
    /// caption it changed under: the caption is set before the phase when
    /// a turn begins, so a change into thinking carries what was asked.
    private var phaseChanges: [PhaseChange] = []
    /// Every change of the user caption, oldest first.
    private var captionChanges: [CaptionChange] = []
    private var observations: Set<AnyCancellable> = []

    /// Every utterance the microphone reported to voice mode, oldest first.
    private var heard: [HeardUtterance] = []
    /// The last of `heard` has had its onset and not yet its final
    /// transcript.
    private var utteranceOpen = false
    /// How many of `heard` earlier rounds have reported.
    private var heardReported = 0
    /// Every spell of the reply being ducked, oldest first.
    private var ducks: [DuckSpell] = []
    private var ducksReported = 0
    /// A sample has been asked for and voice mode has not yet begun the
    /// turn that answers it.
    private var questionPending = false
    /// The microphone was seen not listening, or failed, since the round
    /// began.
    private var microphoneWentDown = false
    private var microphoneFailures: [(at: Date, reason: String)] = []

    private struct PhaseChange {
        let at: Date
        let phase: VoiceModeController.Phase
        let caption: String
    }

    private struct CaptionChange {
        let at: Date
        let text: String
    }

    /// Where the round stood when the microphone heard something.
    private enum Stage: String {
        /// The sample was being asked. The microphone hears the recording
        /// through the speaker, and voice mode drops what it makes of it.
        case question
        /// A turn was under way: its reply was being generated or played.
        case reply
        /// Nothing was under way. Just after a reply, its last words can
        /// still reach the microphone, ringing on in the room.
        case quiet
    }

    /// One utterance the microphone reported to voice mode.
    private struct HeardUtterance {
        /// When its onset was reported.
        let at: Date
        let stage: Stage
        /// The best transcript so far; the final one once `isFinal`.
        var text = ""
        var isFinal = false
        /// Its onset ducked the reply.
        var duckedReply = false
        /// What voice mode took from it as the user's words, as it
        /// captioned them; nil while it has taken nothing.
        var takenAs: String?
    }

    private struct DuckSpell {
        let start: Date
        /// Nil while the reply is still ducked.
        var end: Date?
    }

    /// What one round found, for the summary.
    private struct Round {
        /// The sample's transcript, as voice mode asked it.
        var asked: String?
        var timedOut = false
        /// The microphone listened from the start of the round to its end.
        var microphoneListening = false
        var onsetsDuringReply = 0
        var onsetsWhileQuiet = 0
        /// Utterances outside the question that the recogniser made words
        /// of, and how many of them voice mode took for the user's.
        var echoHeard = 0
        var echoTaken = 0
        var ducks = 0
        var duckedSeconds: TimeInterval = 0
    }

    init(
        chat: ChatController,
        voice: VoiceModeController,
        speech: SpeechOutput,
        microphone: SpeechInput,
        router: AppRouter,
        store: ChatStore,
        rounds: Int
    ) {
        self.chat = chat
        self.voice = voice
        self.speech = speech
        self.microphone = microphone
        self.router = router
        self.store = store
        self.rounds = rounds
    }

    func run() async {
        log.reset()
        observeVoiceMode()
        tap.observe = { [weak self] event, deliver in
            guard let self else { return deliver() }
            self.microphoneReported(event, deliver: deliver)
        }
        log.emit([
            "soak": "begin",
            "date": ISO8601DateFormatter().string(from: started),
            "rounds": rounds,
        ])

        let booted = await wait(Self.bootTimeout) { self.chat.engineState == .ready }
        guard booted else {
            log.emit([
                "soak": "summary",
                "pass": NSNull(),
                "notes": ["engine not ready after \(Int(Self.bootTimeout)) s: \(chat.engineState)"],
            ])
            return
        }
        let bootSeconds = elapsed

        chat.newChat()
        let conversationID = chat.conversation.id
        var setupNotes: [String] = []
        router.isVoiceModePresented = true
        let presented = await wait(Self.presentTimeout) { self.voice.isActive }
        if !presented {
            // The cover's appearance is what starts voice mode; without it
            // the controller is started the way the cover would.
            setupNotes.append("voice mode not active \(Int(Self.presentTimeout)) s after presenting it; started directly")
            voice.start()
        }
        let listening = await wait(Self.microphoneTimeout) { self.microphone.isListening }
        if listening {
            await pause(Self.settleTime)
        } else {
            setupNotes.append("microphone not listening after \(Int(Self.microphoneTimeout)) s; phase \(Self.name(voice.phase))")
        }
        if !voice.hasSampleQuestions {
            setupNotes.append("no sample recordings in this build")
        }
        log.emit([
            "soak": "setup",
            "engine_ready": Self.rounded(bootSeconds),
            "microphone": voice.availability.map { "\($0)" } ?? "not checked",
            "microphone_listening": listening,
            "sample_questions": voice.hasSampleQuestions,
            "notes": setupNotes,
        ])

        var results: [Round] = []
        var notes: [String] = []
        if voice.hasSampleQuestions {
            var known = Set(chat.conversation.messages.map(\.id))
            for number in 1...rounds {
                guard voice.isActive else {
                    notes.append("voice mode ended on its own before round \(number)")
                    break
                }
                results.append(await round(number, conversationID: conversationID, known: &known))
            }
        }

        // Read before voice mode ends: ending it stops a reply still under
        // way, which would then pass for a false barge-in.
        let messages = snapshot(of: conversationID)
        removeTap()
        voice.end()
        router.isVoiceModePresented = false
        summarize(messages, results: results, notes: notes)
        saveTranscript(messages)

        await pause(Self.dismissTime)
        chat.newChat()
        store.delete(conversationID)
        print(String(format: "[soak] finished: %d rounds in %.1f s", results.count, elapsed))
        fflush(stdout)
    }

    /// The whole soak conversation, as voice mode left it, beside the
    /// soak's log: the conversation itself is deleted afterwards so repeated
    /// runs do not fill the sidebar.
    private func saveTranscript(_ messages: [StoredMessage]) {
        let rows: [[String: Any]] = messages.map { message in
            var row: [String: Any] = [
                "role": message.role.rawValue,
                "text": message.text,
                "via_voice": message.viaVoice,
                "stopped": message.wasStopped,
            ]
            if let stats = message.stats {
                row["reused"] = stats.reused
                row["new_prefill"] = stats.newPrefill
                row["generated"] = stats.generated
                row["note"] = stats.note
            }
            return row
        }
        guard let data = try? JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted]) else { return }
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("voice-soak-conversation.json")
        try? data.write(to: url, options: .atomic)
    }

    // MARK: - Rounds

    /// Asks the next sample once voice mode is quiet, waits for the reply
    /// to be spoken and for a further quiet spell, and records what the
    /// round added to the conversation, what the microphone heard and how
    /// often the reply was ducked. `known` holds the messages already
    /// accounted for by earlier rounds.
    private func round(_ number: Int, conversationID: UUID, known: inout Set<UUID>) async -> Round {
        var notes: [String] = []
        var result = Round()
        microphoneWentDown = !microphone.isListening
        let failuresBefore = microphoneFailures.count

        let ready = await wait(Self.quietTimeout) { self.isSettled }
        if !ready {
            // Asking now cuts off whatever is still under way, and a reply
            // cut off that way is marked stopped like a barge-in's.
            notes.append("still busy \(Int(Self.quietTimeout)) s before asking, so asking cut it off: phase \(Self.name(voice.phase)), speaking \(speech.isSpeaking), generating \(chat.isGenerating)")
        }
        if case .failed(let reason) = voice.phase {
            notes.append("voice mode showed a failure before asking: \(reason)")
        }
        let first = phaseChanges.count
        let askedAt = Date()
        questionPending = true
        voice.askSampleQuestion()

        let questionStarted = await wait(Self.questionTimeout) {
            !self.turnStarts(from: first).isEmpty || self.isFailed
        }
        questionPending = false
        let question = turnStarts(from: first).first
        if let question {
            result.asked = phaseChanges[question].caption.trimmingCharacters(in: .whitespacesAndNewlines)
            let replied = await wait(Self.replyTimeout) { self.isSettled }
            if !replied {
                result.timedOut = true
                notes.append("reply not over \(Int(Self.replyTimeout)) s after the question: phase \(Self.name(voice.phase)), speaking \(speech.isSpeaking)")
            } else if case .failed(let reason) = voice.phase {
                notes.append("the turn ended in a failure: \(reason)")
            }
        } else {
            result.timedOut = true
            notes.append(questionStarted
                ? "the sample failed before it was asked: \(Self.name(voice.phase))"
                : "no question \(Int(Self.questionTimeout)) s after the sample was asked for: phase \(Self.name(voice.phase))")
        }

        let quiet = await waitForQuiet()
        if !quiet {
            notes.append("never quiet for \(Int(Self.quietPeriod)) s within \(Int(Self.quietTimeout)) s of the reply")
        }

        // What this round added: everything since the last round's
        // snapshot, so a turn that started in between counts here.
        let messages = snapshot(of: conversationID)
        let added = messages.filter { !known.contains($0.id) }
        known.formUnion(messages.map(\.id))

        var extras = added.filter { $0.role == .user }
        let askedIndex = result.asked.flatMap { asked in extras.firstIndex { Self.trimmed($0.text) == asked } }
        var reply: StoredMessage?
        if let askedIndex {
            let questionID = extras.remove(at: askedIndex).id
            if let position = messages.firstIndex(where: { $0.id == questionID }),
               position + 1 < messages.count,
               messages[position + 1].role == .assistant {
                reply = messages[position + 1]
            } else {
                notes.append("no reply after the question in the conversation")
            }
        } else if result.asked != nil {
            notes.append("the question is not in the conversation as asked")
        }

        let turns = turnStarts(from: first)
        let speakingAt = question.flatMap { index in
            phaseChanges[index...].first { $0.phase == .speaking }?.at
        }
        let listeningAt = question.flatMap { index in
            phaseChanges[index...].first { Self.isListeningOrFailed($0.phase) }?.at
        }

        result.microphoneListening = !microphoneWentDown
        for failure in microphoneFailures[failuresBefore...] {
            notes.append("the microphone failed \(Self.rounded(failure.at.timeIntervalSince(askedAt))) s in: \(failure.reason)")
        }

        // Everything heard since the last round reported, so an utterance
        // that began between rounds counts here.
        let roundHeard = Array(heard[heardReported...])
        heardReported = heard.count
        let echo = roundHeard.filter { $0.stage != .question }
        let echoTranscribed = echo.filter { !$0.text.isEmpty }
        result.onsetsDuringReply = echo.filter { $0.stage == .reply }.count
        result.onsetsWhileQuiet = echo.filter { $0.stage == .quiet }.count
        result.echoHeard = echoTranscribed.count
        result.echoTaken = echoTranscribed.filter { $0.takenAs != nil }.count

        let now = Date()
        let roundDucks = Array(ducks[ducksReported...])
        ducksReported = ducks.count
        if let last = roundDucks.last, last.end == nil {
            notes.append("the reply was still ducked when the round ended")
        }
        result.ducks = roundDucks.count
        result.duckedSeconds = roundDucks.reduce(0) { $0 + ($1.end ?? now).timeIntervalSince($1.start) }

        log.emit([
            "soak": "round",
            "round": number,
            "t": Self.rounded(askedAt.timeIntervalSince(started)),
            "microphone_listening": result.microphoneListening,
            "asked": result.asked ?? NSNull(),
            "reply": reply?.text ?? NSNull(),
            "reply_chars": reply?.text.count ?? 0,
            // What the engine reported for this reply: a reply that only
            // repeats an earlier one shows here as the same token counts.
            "reply_stats": reply?.stats.map { stats in
                [
                    "prompt_tokens": stats.promptTokens,
                    "reused": stats.reused,
                    "new_prefill": stats.newPrefill,
                    "generated": stats.generated,
                    "note": stats.note,
                ] as [String: Any]
            } ?? NSNull(),
            "stopped": reply.map { $0.wasStopped as Any } ?? NSNull(),
            "extra_user_messages": extras.map(\.text),
            "stopped_replies": added.filter { $0.role == .assistant && $0.wasStopped }.map(\.text),
            "turns_started": turns.count,
            "timed_out": result.timedOut,
            "question_finished": Self.seconds(askedAt, question.map { phaseChanges[$0].at }),
            "speaking_started": Self.seconds(askedAt, speakingAt),
            "listening_again": Self.seconds(askedAt, listeningAt),
            "phases": (first..<phaseChanges.count).map { index -> [String: Any] in
                let change = phaseChanges[index]
                var entry: [String: Any] = [
                    "t": Self.rounded(change.at.timeIntervalSince(askedAt)),
                    "phase": Self.name(change.phase),
                ]
                if turns.contains(index) { entry["question"] = change.caption }
                return entry
            },
            // Every utterance the microphone reported, including those
            // voice mode judged to be the reply's echo and never captioned:
            // `taken_as` is null for those, and otherwise holds the words
            // voice mode took for the user's.
            "heard": roundHeard.map { utterance -> [String: Any] in
                [
                    "t": Self.rounded(utterance.at.timeIntervalSince(askedAt)),
                    "during": utterance.stage.rawValue,
                    "text": utterance.text,
                    "final": utterance.isFinal,
                    "ducked_reply": utterance.duckedReply,
                    "taken_as": utterance.takenAs ?? NSNull(),
                ]
            },
            "onsets_during_reply": result.onsetsDuringReply,
            "onsets_while_quiet": result.onsetsWhileQuiet,
            "echo_heard": result.echoHeard,
            "echo_taken": result.echoTaken,
            "ducks": roundDucks.map { spell -> [String: Any] in
                [
                    "t": Self.rounded(spell.start.timeIntervalSince(askedAt)),
                    "seconds": Self.rounded((spell.end ?? now).timeIntervalSince(spell.start)),
                ]
            },
            "ducked_seconds": Self.rounded(result.duckedSeconds),
            "notes": notes,
        ])
        return result
    }

    /// Checks the whole conversation: every user message must be one
    /// round's sample transcript, each used once, and no reply may have
    /// been cut off.
    private func summarize(_ messages: [StoredMessage], results: [Round], notes: [String]) {
        var unclaimed = results.compactMap(\.asked)
        var extras: [String] = []
        for message in messages where message.role == .user {
            if let index = unclaimed.firstIndex(of: Self.trimmed(message.text)) {
                unclaimed.remove(at: index)
            } else {
                extras.append(message.text)
            }
        }
        let stopped = messages.filter { $0.role == .assistant && $0.wasStopped }.map(\.text)
        let timedOut = results.filter(\.timedOut).count
        let unasked = results.filter { $0.asked == nil }.count
        let withoutMicrophone = results.filter { !$0.microphoneListening }.count
        let onsetsDuringReplies = results.reduce(0) { $0 + $1.onsetsDuringReply }
        let onsetsWhileQuiet = results.reduce(0) { $0 + $1.onsetsWhileQuiet }
        let echoHeard = results.reduce(0) { $0 + $1.echoHeard }
        let echoTaken = results.reduce(0) { $0 + $1.echoTaken }

        let clean = extras.isEmpty && stopped.isEmpty && unclaimed.isEmpty && timedOut == 0 && unasked == 0
            && results.count == rounds
        // A round that ran without the microphone could not have answered
        // itself, so a clean run shows nothing either way unless the
        // microphone listened throughout every round.
        let pass: Any
        let result: String
        if !clean {
            pass = false
            result = "fail"
        } else if withoutMicrophone > 0 {
            pass = NSNull()
            result = "inconclusive, the microphone was not listening throughout every round"
        } else {
            pass = true
            if echoHeard > 0 {
                result = "pass, echo transcribed and filtered"
            } else if onsetsDuringReplies + onsetsWhileQuiet > 0 {
                result = "pass, echo set off the detector but was never transcribed"
            } else {
                result = "pass, no echo reached the detector"
            }
        }

        log.emit([
            "soak": "summary",
            "pass": pass,
            "result": result,
            "rounds": rounds,
            "rounds_run": results.count,
            "rounds_timed_out": timedOut,
            "rounds_without_question": unasked,
            "rounds_without_microphone": withoutMicrophone,
            "user_messages": messages.filter { $0.role == .user }.count,
            "assistant_messages": messages.filter { $0.role == .assistant }.count,
            "extra_user_message_count": extras.count,
            "extra_user_messages": extras,
            "stopped_reply_count": stopped.count,
            "stopped_replies": stopped,
            "questions_missing": unclaimed,
            "onsets_during_replies": onsetsDuringReplies,
            "onsets_while_quiet": onsetsWhileQuiet,
            "echo_heard": echoHeard,
            "echo_filtered": echoHeard - echoTaken,
            "echo_taken": echoTaken,
            "ducks": results.reduce(0) { $0 + $1.ducks },
            "ducked_seconds": Self.rounded(results.reduce(0) { $0 + $1.duckedSeconds }),
            "seconds": Self.rounded(elapsed),
            "notes": notes,
        ])
    }

    // MARK: - Watching voice mode

    private func observeVoiceMode() {
        // Both publishers fire as the value is about to change, on the
        // main thread where voice mode sets it, so the caption read with a
        // phase change is the one the new phase starts under.
        voice.$phase
            .removeDuplicates()
            .sink { [weak self] phase in
                MainActor.assumeIsolated { self?.record(phase) }
            }
            .store(in: &observations)
        voice.$userCaption
            .removeDuplicates()
            .sink { [weak self] text in
                MainActor.assumeIsolated {
                    self?.captionChanges.append(CaptionChange(at: Date(), text: text))
                }
            }
            .store(in: &observations)
    }

    private func record(_ phase: VoiceModeController.Phase) {
        phaseChanges.append(PhaseChange(at: Date(), phase: phase, caption: voice.userCaption))
        if phase == .thinking { questionPending = false }
        // A turn ending clears the duck before its phase changes, so the
        // spell's end is read here rather than at the next poll.
        checkDuck()
    }

    /// One of the microphone's calls to voice mode, passed on by
    /// `deliver`. Where the round stood is read before voice mode hears of
    /// it, and what voice mode did with it afterwards: an utterance it
    /// took for the user's changes the caption or begins a turn, and an
    /// onset over a reply ducks it.
    private func microphoneReported(_ event: MicrophoneTap.Event, deliver: () -> Void) {
        let captionsBefore = captionChanges.count
        let phasesBefore = phaseChanges.count
        let wasDucked = speech.isDucked
        if case .onset = event {
            heard.append(HeardUtterance(at: Date(), stage: stage))
            utteranceOpen = true
        }
        deliver()
        checkDuck()

        switch event {
        case .onset:
            if !wasDucked, speech.isDucked { heard[heard.count - 1].duckedReply = true }
        case .partial(let text), .finalized(let text):
            guard utteranceOpen, !heard.isEmpty else { return }
            let last = heard.count - 1
            if !text.isEmpty { heard[last].text = text }
            if case .finalized = event {
                heard[last].isFinal = true
                utteranceOpen = false
            }
            if let caption = captionChanges[captionsBefore...].last(where: { !$0.text.isEmpty }) {
                heard[last].takenAs = caption.text
            } else if heard[last].takenAs == nil, !turnStarts(from: phasesBefore).isEmpty {
                // Asked as it was already captioned, so the caption did not
                // change.
                heard[last].takenAs = voice.userCaption
            }
        case .failed(let reason):
            utteranceOpen = false
            microphoneWentDown = true
            microphoneFailures.append((at: Date(), reason: reason))
        }
    }

    /// Where the round stands now.
    private var stage: Stage {
        if questionPending { return .question }
        return isSettled ? .quiet : .reply
    }

    /// Notes the start or the end of a spell of the reply being ducked.
    private func checkDuck() {
        let spellOpen = ducks.last.map { $0.end == nil } ?? false
        if speech.isDucked, !spellOpen {
            ducks.append(DuckSpell(start: Date()))
        } else if !speech.isDucked, spellOpen {
            ducks[ducks.count - 1].end = Date()
        }
    }

    /// Puts the tap between the microphone and voice mode. Voice mode
    /// makes itself the microphone's delegate each time it opens a
    /// session, which during a soak it does again only after the last
    /// session failed, so the tap is put back on the next poll.
    private func installTap() {
        guard voice.isActive, let current = microphone.delegate, current !== tap else { return }
        tap.downstream = current
        microphone.delegate = tap
        // A new session: the last one never finalised what it had open.
        utteranceOpen = false
    }

    private func removeTap() {
        if microphone.delegate === tap { microphone.delegate = tap.downstream }
    }

    /// Runs on every poll.
    private func tick() {
        checkDuck()
        if !microphone.isListening {
            microphoneWentDown = true
            // A session that has ended never finalises what it had open.
            utteranceOpen = false
        }
        installTap()
    }

    /// The phase changes from `start` on that began a turn. Only a new
    /// question moves voice mode into thinking from anything but
    /// speaking; from speaking it is a reply waiting on the model.
    private func turnStarts(from start: Int) -> [Int] {
        (start..<phaseChanges.count).filter { index in
            phaseChanges[index].phase == .thinking
                && (index == 0 || phaseChanges[index - 1].phase != .speaking)
        }
    }

    /// Nothing under way: no reply generating or playing, and voice mode
    /// back to listening, or showing why it cannot.
    private var isSettled: Bool {
        !speech.isSpeaking && !chat.isGenerating && Self.isListeningOrFailed(voice.phase)
    }

    private var isFailed: Bool {
        if case .failed = voice.phase { return true }
        return false
    }

    /// Waits for voice mode to stay settled, with no change of phase or
    /// caption and no utterance open on the microphone, for
    /// `quietPeriod`. A caption changing with no turn under way is the
    /// microphone transcribing something, so it restarts the count as a
    /// new turn would. An utterance still open could yet become a turn,
    /// and the next sample would cut it off before voice mode judged it.
    private func waitForQuiet() async -> Bool {
        let deadline = Date().addingTimeInterval(Self.quietTimeout)
        var quietSince = Date()
        while Date() < deadline {
            tick()
            let now = Date()
            if !isSettled || utteranceOpen { quietSince = now }
            if let change = phaseChanges.last?.at, change > quietSince { quietSince = change }
            if let change = captionChanges.last?.at, change > quietSince { quietSince = change }
            if now.timeIntervalSince(quietSince) >= Self.quietPeriod { return true }
            await pause(0.1)
        }
        return false
    }

    /// The soak conversation's messages as they stand.
    private func snapshot(of id: UUID) -> [StoredMessage] {
        if chat.conversation.id == id { return chat.conversation.messages }
        return store.conversation(id)?.messages ?? []
    }

    // MARK: - Waiting

    private var elapsed: TimeInterval { Date().timeIntervalSince(started) }

    /// Polls `condition` every 100 ms until it holds or `timeout` passes.
    /// Returns whether it held.
    private func wait(_ timeout: TimeInterval, until condition: () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while true {
            tick()
            if condition() { return true }
            if Date() >= deadline { return false }
            await pause(0.1)
        }
    }

    private func pause(_ seconds: TimeInterval) async {
        try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
    }

    // MARK: - Formatting

    private static func isListeningOrFailed(_ phase: VoiceModeController.Phase) -> Bool {
        if case .failed = phase { return true }
        return phase == .listening
    }

    private static func name(_ phase: VoiceModeController.Phase) -> String {
        if case .failed(let reason) = phase { return "failed: \(reason)" }
        return "\(phase)"
    }

    private static func trimmed(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Hundredths of a second, as a decimal: a binary double such as 0.3
    /// would be written out as 0.29999999999999999.
    private static func rounded(_ value: TimeInterval) -> Decimal {
        Decimal(Int((value * 100).rounded())) / 100
    }

    private static func seconds(_ from: Date, _ to: Date?) -> Any {
        guard let to else { return NSNull() }
        return rounded(to.timeIntervalSince(from))
    }
}

/// Stands between the microphone and voice mode and passes every call
/// straight on, so the soak sees what the microphone heard whether or not
/// voice mode took it for the user. Calls arrive on the main queue, as the
/// contract has it, and each is passed on within the same call, so voice
/// mode hears them as it would without the tap.
@MainActor
private final class MicrophoneTap: @preconcurrency SpeechInputDelegate {
    enum Event {
        case onset
        case partial(String)
        case finalized(String)
        case failed(String)
    }

    /// Voice mode's own delegate.
    weak var downstream: SpeechInputDelegate?
    /// Given each event and the call that passes it on, which it makes
    /// exactly once.
    var observe: (Event, () -> Void) -> Void = { _, deliver in deliver() }

    func speechInputDidChangeAvailability(_ availability: VoiceInputAvailability) {
        downstream?.speechInputDidChangeAvailability(availability)
    }

    func speechInputDidUpdatePartial(_ text: String) {
        observe(.partial(text)) { self.downstream?.speechInputDidUpdatePartial(text) }
    }

    func speechInputDidFinalize(_ text: String) {
        observe(.finalized(text)) { self.downstream?.speechInputDidFinalize(text) }
    }

    func speechInputDidUpdateLevel(_ level: Float) {
        downstream?.speechInputDidUpdateLevel(level)
    }

    func speechInputDidDetectSpeechOnset() {
        observe(.onset) { self.downstream?.speechInputDidDetectSpeechOnset() }
    }

    func speechInputDidFail(_ error: Error) {
        observe(.failed(error.localizedDescription)) { self.downstream?.speechInputDidFail(error) }
    }
}

/// Writes each line to stdout as `PIESOAK {json}` and appends it to
/// Documents/voice-soak.jsonl, which starts empty on every run.
private final class SoakLog {
    private let url = FileManager.default
        .urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("voice-soak.jsonl")

    func reset() {
        try? FileManager.default.removeItem(at: url)
    }

    func emit(_ object: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]),
              let line = String(data: data, encoding: .utf8)
        else {
            print("PIESOAK {\"soak\":\"log\",\"error\":\"unserialisable line\"}")
            fflush(stdout)
            return
        }
        print("PIESOAK \(line)")
        fflush(stdout)

        let bytes = Data((line + "\n").utf8)
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: bytes)
        } else {
            try? bytes.write(to: url)
        }
    }
}
