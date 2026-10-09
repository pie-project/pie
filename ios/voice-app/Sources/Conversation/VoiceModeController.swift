import Foundation

/// Full-screen voice mode: listen, think, speak, repeat, with the user
/// able to talk over a reply to interrupt it.
///
/// The microphone runs one `.conversation` session for the whole of voice
/// mode, sample questions included. Its echo canceller keeps the reply
/// coming out of the speaker from sounding like the user, so the session
/// stays live while a reply is generated and spoken, and the user can talk
/// over it without a tap.
///
/// The detector's onset (`speechInputDidDetectSpeechOnset`) only means
/// something loud started, so talking over a reply goes in two steps:
///   - at onset the reply is held: ducked at once if it is playing, so the
///     user is not drowned out, with anything more it says kept back,
///     while the engine carries on generating;
///   - the first recognised words that are not the reply's own make it a
///     barge-in: the reply is cancelled and what the user is saying
///     becomes the next question. The echo canceller is good but not
///     perfect, and what leaks through it is the reply's own words, so a
///     transcript made of them is the reply heard back, not the user. An
///     utterance that ends with no words of the user's in it (a cough, a
///     chair, the room, the reply's echo) releases the hold and the reply
///     carries on at full volume, never cut off or repeated.
///
/// A pause in the middle of a question ends an utterance like any other,
/// and the turn it starts would answer half a question. When the user
/// carries on talking before any of that turn's reply has played, the turn
/// is withdrawn and its words are asked again with what they say next.
///
/// Every reply belongs to a numbered turn. Cancelling retires the turn, so
/// text the engine produced before the cancel landed is dropped rather
/// than spoken over the next question.
@MainActor
final class VoiceModeController: ObservableObject {

    enum Phase: Equatable {
        case idle
        case listening
        case thinking
        case speaking
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var isActive = false
    @Published private(set) var isMuted = false
    /// 0...1, from the microphone.
    @Published private(set) var inputLevel: Float = 0
    /// 0...1, from what is being spoken.
    @Published private(set) var outputLevel: Float = 0
    /// What the user is saying or just said.
    @Published private(set) var userCaption = ""
    /// The reply as it streams.
    @Published private(set) var assistantCaption = ""
    @Published private(set) var availability: VoiceInputAvailability?

    /// Bundled sample questions exist.
    var hasSampleQuestions: Bool {
        guard let sample else { return false }
        return (sample as? SampleQuestionInput)?.hasRecordings ?? true
    }

    private let chat: ChatController
    private let microphone: SpeechInput
    private let sample: SpeechInput?
    private let speech: SpeechOutput
    private let settings: AppSettings
    private let chunker = SentenceChunker()

    /// The microphone and the sample report through the same delegate
    /// protocol; a relay per source tells them apart.
    private let microphoneRelay: InputRelay
    private let sampleRelay: InputRelay

    /// Bumped by every `start()` and `end()`, so a permission check that
    /// returns after voice mode was left (or left and entered again) does
    /// nothing.
    private var visit: UInt64 = 0
    private var turnCounter: UInt64 = 0
    /// The turn whose reply is still welcome; nil between turns.
    private var activeTurn: UInt64?
    /// What the active turn asked.
    private var turnQuestion = ""
    /// The active turn's reply is still being generated.
    private var isGenerating = false
    /// The active turn has handed the synthesizer at least one sentence.
    private var hasSpoken = false
    /// Some of the active turn's reply has been audible.
    private var replyWasHeard = false
    /// The speech meter level above which a reply counts as audible.
    private static let audibleLevel: Float = 0.05
    /// Every sentence of the active turn's reply, in the form it is spoken.
    private var replySentences: [String] = []
    /// How many of `replySentences` the synthesizer has been handed.
    private var handedToSpeech = 0

    /// The user may be talking over the active turn; see `holdReply()`.
    private var isHoldingReply = false
    /// Bumped by every hold and every end of one, so a timeout that fires
    /// after its hold ended does nothing.
    private var holdCounter: UInt64 = 0
    /// How long a hold waits for a recognised word before deciding the
    /// sound that started it was not the user: a noise that goes on
    /// without words (a fan switching on, traffic) would otherwise keep
    /// the reply silent until the detector's utterance cap.
    private static let holdTimeout: UInt64 = 3_000_000_000

    /// A question withdrawn because the user carried on talking before its
    /// reply played; asked again together with what they say next.
    private var unfinishedQuestion: String?

    /// The microphone has reported the onset of an utterance it has not
    /// finalised yet.
    private var microphoneUtteranceOpen = false
    /// That utterance is not a question: it began while a sample recording
    /// was playing, so it is what the microphone heard of the recording, or
    /// it was cut short to make way for one.
    private var discardsMicrophoneUtterance = false

    /// A bundled recording is playing in place of the user.
    private var isSamplePlaying = false
    private var sampleRequest: UInt64 = 0
    /// Why the microphone stopped working, until it is restarted. A turn
    /// that ends while this is set ends in `.failed` rather than
    /// `.listening`, so the orb offers the tap that restarts it.
    private var inputFailure: String?

    init(chat: ChatController, microphone: SpeechInput, sample: SpeechInput?, speech: SpeechOutput, settings: AppSettings) {
        self.chat = chat
        self.microphone = microphone
        self.sample = sample
        self.speech = speech
        self.settings = settings
        microphoneRelay = InputRelay(source: .microphone)
        sampleRelay = InputRelay(source: .sample)
        microphoneRelay.owner = self
        sampleRelay.owner = self
    }

    // MARK: - Intents

    /// Enters voice mode on the open conversation and starts listening.
    func start() {
        guard !isActive else { return }
        visit += 1
        chat.stopReadingAloud()
        isActive = true
        isMuted = false
        inputFailure = nil
        unfinishedQuestion = nil
        phase = .listening
        userCaption = ""
        assistantCaption = ""
        inputLevel = 0
        outputLevel = 0

        speech.delegate = self
        speech.voiceIdentifier = settings.voiceIdentifier
        speech.rate = settings.speechRate
        // A session left running elsewhere (the composer's dictation) is
        // ended rather than adopted: voice mode starts its own
        // echo-cancelled one.
        if microphone.isListening { microphone.cancel() }
        forgetMicrophoneUtterance()
        microphone.delegate = microphoneRelay
        prepareAndListen()
    }

    /// Leaves voice mode; the turns stay in the conversation.
    func end() {
        guard isActive else { return }
        visit += 1
        interrupt()
        unfinishedQuestion = nil
        microphone.cancel()
        forgetMicrophoneUtterance()
        if isSamplePlaying {
            sample?.cancel()
            isSamplePlaying = false
        }
        if speech.isSpeaking { speech.stop() }
        isActive = false
        isMuted = false
        phase = .idle
        inputLevel = 0
        outputLevel = 0
    }

    /// Speaking or thinking: interrupt and listen. Listening: finish the
    /// utterance now. Idle or failed: listen.
    func tapOrb() {
        guard isActive else {
            start()
            return
        }
        switch phase {
        case .thinking, .speaking:
            interrupt()
        case .listening:
            if isSamplePlaying {
                // Cuts the recording short and asks what it had said.
                sample?.stop()
            } else if isMuted {
                toggleMute()
            } else {
                // In a conversation session this finalises what has been
                // said so far and keeps the session listening.
                microphone.stop()
            }
        case .idle, .failed:
            phase = .listening
            userCaption = ""
            inputFailure = nil
            if !microphone.isListening { prepareAndListen() }
        }
    }

    func toggleMute() {
        guard isActive else { return }
        isMuted.toggle()
        if isMuted {
            // No session, so no barge-in either: a muted user can let a
            // reply play out whatever the room sounds like.
            microphone.cancel()
            forgetMicrophoneUtterance()
            inputLevel = 0
            releaseHold()
            if activeTurn == nil { userCaption = "" }
            askUnfinishedQuestion()
        } else {
            startMicrophone()
        }
    }

    /// Plays the next bundled sample question and answers it, waiting for
    /// the recording to finish before the reply starts.
    ///
    /// The microphone stays open throughout. The recording plays through
    /// the same echo-cancelled engine as the replies, so it joins the
    /// running engine with no rebuild and no change of audio route, and the
    /// reply to it can be talked over from its first word, against an echo
    /// canceller and a detector that have already settled. What the
    /// microphone picks up while the recording plays is dropped.
    func askSampleQuestion() {
        guard isActive, let sample else { return }
        interrupt()
        unfinishedQuestion = nil
        if isSamplePlaying { sample.cancel() }
        // An utterance under way is replaced by the recording.
        if microphoneUtteranceOpen {
            discardsMicrophoneUtterance = true
            microphone.stop()
        }
        isSamplePlaying = true
        sampleRequest += 1
        let request = sampleRequest
        phase = .listening
        userCaption = ""
        assistantCaption = ""
        inputLevel = 0
        sample.delegate = sampleRelay

        Task {
            let availability = await sample.prepare()
            guard isActive, isSamplePlaying, request == sampleRequest else { return }
            guard availability.isReady else {
                isSamplePlaying = false
                phase = .failed(Self.reason(availability))
                startMicrophone()
                return
            }
            do {
                try sample.start(mode: .conversation)
            } catch {
                isSamplePlaying = false
                phase = .failed(error.localizedDescription)
                startMicrophone()
            }
        }
    }

    // MARK: - Microphone

    /// Re-checks availability before listening: speech assets finish
    /// downloading and permissions get granted in Settings, so the answer
    /// from the last check is not the answer now.
    private func prepareAndListen() {
        let current = visit
        Task {
            let availability = await microphone.prepare()
            guard isActive, visit == current else { return }
            self.availability = availability
            if availability.isReady {
                startMicrophone()
            } else {
                inputFailure = Self.reason(availability)
                if activeTurn == nil { phase = .failed(Self.reason(availability)) }
            }
        }
    }

    /// Opens the session if it is not open. A recording that starts before
    /// the microphone is up (the permission check is still out) leaves it
    /// to the recording's end.
    private func startMicrophone() {
        guard isActive, !isMuted, !isSamplePlaying, !microphone.isListening else { return }
        microphone.delegate = microphoneRelay
        do {
            try microphone.start(mode: .conversation)
            forgetMicrophoneUtterance()
            inputFailure = nil
        } catch {
            inputFailure = error.localizedDescription
            if activeTurn == nil { phase = .failed(error.localizedDescription) }
        }
    }

    /// The session ended or started afresh: no utterance is open.
    private func forgetMicrophoneUtterance() {
        microphoneUtteranceOpen = false
        discardsMicrophoneUtterance = false
    }

    // MARK: - Turns

    private func beginTurn(_ utterance: String) {
        userCaption = utterance
        assistantCaption = ""
        switch chat.engineState {
        case .ready:
            break
        case .booting:
            phase = .failed("Pie is still loading the model. Ask again in a moment.")
            return
        case .failed(let reason):
            phase = .failed("The model failed to load: \(reason)")
            return
        }

        turnCounter += 1
        let turn = turnCounter
        activeTurn = turn
        turnQuestion = utterance
        isGenerating = true
        hasSpoken = false
        replyWasHeard = false
        replySentences = []
        handedToSpeech = 0
        isHoldingReply = false
        chunker.reset()
        phase = .thinking

        Task {
            // Interrupted before the reply was even asked for: asking now
            // would only queue a reply nobody wants ahead of the next one.
            guard activeTurn == turn else { return }
            let message = await chat.sendSpoken(utterance) { [weak self] delta in
                self?.receive(delta, turn: turn)
            }
            finishGenerating(turn: turn, message: message)
        }
    }

    /// One chunk of reply text: captioned at once, spoken a sentence at a
    /// time so speech overlaps the rest of the generation.
    private func receive(_ delta: String, turn: UInt64) {
        guard activeTurn == turn else { return }
        assistantCaption += delta
        for sentence in chunker.push(delta) {
            speak(sentence)
        }
    }

    private func speak(_ sentence: String) {
        // The voice prompt asks for plain speech, but a small model still
        // slips into Markdown now and then.
        let spoken = MessageRendering.speakable(sentence)
        guard !spoken.isEmpty else { return }
        replySentences.append(spoken)
        guard !isHoldingReply else { return }
        handToSpeech(from: replySentences.count - 1)
    }

    /// Hands the synthesizer the reply's sentences from `start` on.
    private func handToSpeech(from start: Int) {
        guard start < replySentences.count else { return }
        for sentence in replySentences[start...] {
            speech.enqueue(sentence)
        }
        handedToSpeech = replySentences.count
        hasSpoken = true
    }

    private func finishGenerating(turn: UInt64, message: StoredMessage?) {
        guard activeTurn == turn else { return }
        isGenerating = false
        if let rest = chunker.flush() {
            speak(rest)
        }
        guard let message, !message.wasStopped else {
            // Failed (the chat's banner says why) or stopped from outside
            // voice mode.
            let reason = chat.banner
            endTurn()
            // Unconditionally: it also closes a speech turn that never
            // managed to start, which would otherwise swallow the next one.
            speech.stop()
            phase = message == nil ? .failed(reason ?? "Pie couldn't answer that.") : afterTurn
            return
        }
        // Whether the rest is said waits on whether the user is talking.
        guard !isHoldingReply else { return }
        speech.finishTurn()
        if !hasSpoken || !speech.isSpeaking {
            endTurn()
            phase = afterTurn
        }
    }

    /// Where voice mode goes when a turn is over.
    private var afterTurn: Phase {
        inputFailure.map(Phase.failed) ?? .listening
    }

    /// Retires the active turn: nothing it still produces is captioned or
    /// spoken.
    private func endTurn() {
        activeTurn = nil
        isGenerating = false
        isHoldingReply = false
        holdCounter += 1
        replySentences = []
        handedToSpeech = 0
        speech.isDucked = false
    }

    /// Silences and cancels the reply in progress, if there is one, and
    /// goes back to listening.
    private func interrupt() {
        guard activeTurn != nil else { return }
        endTurn()
        chunker.reset()
        speech.stop()
        chat.stop()
        outputLevel = 0
        userCaption = ""
        phase = afterTurn
    }

    /// The user may be starting to talk over the active turn. Until words
    /// of theirs are recognised it could as well be a cough, or the reply's
    /// own echo, so the reply is held, not cancelled: ducked if it is
    /// playing, with anything more it says kept back. The engine carries
    /// on, so a false alarm costs a moment of quieter speech rather than
    /// the answer.
    private func holdReply() {
        guard activeTurn != nil, !isHoldingReply else { return }
        isHoldingReply = true
        holdCounter += 1
        let hold = holdCounter
        speech.isDucked = true
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: Self.holdTimeout)
            guard let self, self.holdCounter == hold else { return }
            self.releaseHold()
        }
    }

    /// What started the hold was not the user talking over the reply, so
    /// the reply goes on at full volume, with the sentences kept back
    /// during the hold.
    private func releaseHold() {
        guard isHoldingReply else { return }
        isHoldingReply = false
        holdCounter += 1
        speech.isDucked = false
        guard activeTurn != nil else { return }

        handToSpeech(from: handedToSpeech)
        if !isGenerating {
            // The generation finished during the hold, which kept back the
            // end of the speech turn; its finish report now ends the turn.
            speech.finishTurn()
            guard speech.isSpeaking else {
                endTurn()
                phase = afterTurn
                return
            }
        }
        phase = speech.isSpeaking ? .speaking : .thinking
    }

    /// Whether what the recogniser heard during a reply is the user or the
    /// reply itself, leaking past the echo canceller. The leak is the
    /// reply's own words, so a transcript made almost entirely of them is
    /// the reply. One or two words that all occur in the reply could be
    /// either and wait for more.
    private enum Heard {
        case user
        case echo
        case undecided
    }

    private func classify(_ transcript: String) -> Heard {
        let heard = Self.words(transcript)
        guard !heard.isEmpty else { return .undecided }
        let reply = Set(replySentences.flatMap(Self.words))
        let matched = heard.filter(reply.contains).count
        if matched == heard.count, heard.count < Self.echoDecisionWords { return .undecided }
        return Double(matched) >= Self.echoWordShare * Double(heard.count) ? .echo : .user
    }

    /// Words below which a transcript made entirely of the reply's words
    /// is not yet taken for its echo.
    private static let echoDecisionWords = 3
    /// The share of a transcript's words that must occur in the reply for
    /// it to be the reply's echo.
    private static let echoWordShare = 0.75

    private static func words(_ text: String) -> [String] {
        text.lowercased()
            .split { !$0.isLetter && !$0.isNumber && $0 != "'" }
            .map(String.init)
    }

    /// A word was recognised while a turn was under way: the user is
    /// talking over it, and what they are saying is the next question.
    private func userTalkedOver() {
        guard activeTurn != nil else { return }
        let question = turnQuestion
        let heardReply = replyWasHeard
        interrupt()
        guard !heardReply else { return }
        // None of the reply was heard, so the user is not answering it:
        // they paused mid-question and carried on.
        chat.withdrawSpokenExchange(asking: question)
        unfinishedQuestion = question
    }

    /// The rest of a question that was cut by a pause is not coming (the
    /// microphone was muted or failed): the question as it stood is asked.
    private func askUnfinishedQuestion() {
        guard let question = unfinishedQuestion, activeTurn == nil, !isSamplePlaying else { return }
        unfinishedQuestion = nil
        beginTurn(question)
    }

    private static func joined(_ first: String?, _ second: String) -> String {
        let rest = second.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first, !first.isEmpty else { return rest }
        return rest.isEmpty ? first : first + " " + rest
    }

    private static func reason(_ availability: VoiceInputAvailability) -> String {
        switch availability {
        case .denied(let reason), .unavailable(let reason): return reason
        case .ready: return ""
        }
    }

    // MARK: - Input events

    fileprivate func inputDidChangeAvailability(_ availability: VoiceInputAvailability, from source: InputRelay.Source) {
        guard source == .microphone else { return }
        self.availability = availability
        guard isActive else { return }
        if availability.isReady {
            // The recogniser came back under a session that never ended.
            guard inputFailure != nil, microphone.isListening else { return }
            inputFailure = nil
            if activeTurn == nil { phase = .listening }
        } else {
            inputFailure = Self.reason(availability)
            if activeTurn == nil { phase = .failed(Self.reason(availability)) }
        }
    }

    fileprivate func inputDidUpdatePartial(_ text: String, from source: InputRelay.Source) {
        guard isActive else { return }
        switch source {
        case .sample:
            guard isSamplePlaying else { return }
            userCaption = text
        case .microphone:
            guard !isSamplePlaying, !discardsMicrophoneUtterance else { return }
            if activeTurn != nil {
                // The reply's own words, heard back, are neither a barge-in
                // nor a caption.
                guard classify(text) == .user else { return }
                userTalkedOver()
            }
            userCaption = Self.joined(unfinishedQuestion, text)
        }
    }

    fileprivate func inputDidFinalize(_ text: String, from source: InputRelay.Source) {
        guard isActive else { return }
        let utterance = text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch source {
        case .sample:
            guard isSamplePlaying else { return }
            isSamplePlaying = false
            inputLevel = 0
            // A sample that keeps its session open after the recording
            // would otherwise hold the audio route the microphone needs.
            if sample?.isListening == true { sample?.cancel() }
            // What the microphone heard of the recording is still an open
            // utterance; ended now, so the user's first word over the reply
            // starts one of its own.
            if microphoneUtteranceOpen, discardsMicrophoneUtterance { microphone.stop() }
            startMicrophone()
            guard !utterance.isEmpty else {
                if activeTurn == nil { userCaption = "" }
                return
            }
            interrupt()
            beginTurn(utterance)

        case .microphone:
            let discarded = discardsMicrophoneUtterance
            forgetMicrophoneUtterance()
            guard !discarded, !isSamplePlaying else { return }
            if activeTurn != nil {
                guard classify(utterance) == .user else {
                    // A cough, a chair, the room, the reply's own echo:
                    // nothing to answer, and the reply goes on.
                    releaseHold()
                    return
                }
                userTalkedOver()
            }
            let question = Self.joined(unfinishedQuestion, utterance)
            unfinishedQuestion = nil
            guard !question.isEmpty else {
                userCaption = ""
                return
            }
            beginTurn(question)
        }
    }

    fileprivate func inputDidUpdateLevel(_ level: Float, from source: InputRelay.Source) {
        guard isActive else { return }
        switch source {
        case .microphone:
            guard !isSamplePlaying else { return }
            inputLevel = isMuted ? 0 : level
        case .sample:
            inputLevel = level
        }
    }

    fileprivate func inputDidDetectOnset(from source: InputRelay.Source) {
        // The sample is the question itself; it never interrupts anything.
        guard isActive, source == .microphone else { return }
        microphoneUtteranceOpen = true
        if isSamplePlaying {
            discardsMicrophoneUtterance = true
            return
        }
        if activeTurn != nil {
            holdReply()
        } else {
            userCaption = unfinishedQuestion ?? ""
            phase = .listening
        }
    }

    fileprivate func inputDidFail(_ error: Error, from source: InputRelay.Source) {
        guard isActive else { return }
        inputLevel = 0
        switch source {
        case .sample:
            guard isSamplePlaying else { return }
            isSamplePlaying = false
            phase = .failed(error.localizedDescription)
            startMicrophone()
        case .microphone:
            // The session is over, and nobody can be talking over the
            // reply through it.
            forgetMicrophoneUtterance()
            releaseHold()
            inputFailure = error.localizedDescription
            if activeTurn == nil { phase = .failed(error.localizedDescription) }
            askUnfinishedQuestion()
        }
    }
}

// MARK: - SpeechOutputDelegate

extension VoiceModeController: @preconcurrency SpeechOutputDelegate {
    func speechOutputDidStart() {
        // A start reported after a hold already silenced it is not one.
        guard activeTurn != nil, !isHoldingReply, phase == .thinking else { return }
        phase = .speaking
    }

    func speechOutputDidFinish(interrupted: Bool) {
        outputLevel = 0
        // A hold settles the turn itself when it ends, and its own stop
        // reports here too.
        guard isActive, activeTurn != nil, !isHoldingReply else { return }
        if interrupted {
            // Voice mode's own stops retire the turn or start a hold before
            // they silence the speech, so a cut-off report for the turn
            // still being spoken came from elsewhere: a call or Siri took
            // the audio, or the engine could not play. The reply can no
            // longer be heard; it is stopped like a tap would stop it,
            // rather than left showing "Speaking" with nothing coming out.
            // A late report for a turn already retired finds the next turn
            // thinking, or speaking again, and is ignored.
            if phase == .speaking, !speech.isSpeaking {
                interrupt()
            }
            return
        }
        if isGenerating {
            // Speech caught up with the model; more sentences are coming.
            phase = .thinking
        } else {
            endTurn()
            phase = afterTurn
        }
    }

    func speechOutputLevel(_ level: Float) {
        outputLevel = level
        // Heard, not merely queued: the start is reported before the
        // synthesizer has rendered a sound.
        if level > Self.audibleLevel, activeTurn != nil, !isHoldingReply {
            replyWasHeard = true
        }
    }
}

/// Forwards one input's delegate callbacks to voice mode, tagged with
/// which input sent them.
@MainActor
private final class InputRelay: @preconcurrency SpeechInputDelegate {
    enum Source {
        case microphone
        case sample
    }

    let source: Source
    weak var owner: VoiceModeController?

    init(source: Source) {
        self.source = source
    }

    func speechInputDidChangeAvailability(_ availability: VoiceInputAvailability) {
        owner?.inputDidChangeAvailability(availability, from: source)
    }

    func speechInputDidUpdatePartial(_ text: String) {
        owner?.inputDidUpdatePartial(text, from: source)
    }

    func speechInputDidFinalize(_ text: String) {
        owner?.inputDidFinalize(text, from: source)
    }

    func speechInputDidUpdateLevel(_ level: Float) {
        owner?.inputDidUpdateLevel(level, from: source)
    }

    func speechInputDidDetectSpeechOnset() {
        owner?.inputDidDetectOnset(from: source)
    }

    func speechInputDidFail(_ error: Error) {
        owner?.inputDidFail(error, from: source)
    }
}
