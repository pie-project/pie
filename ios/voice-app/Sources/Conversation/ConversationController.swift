import Foundation

/// Drives one spoken conversation: listen, answer, speak, repeat.
///
/// This is the only type that holds all three layers at once, and it
/// holds them as protocols — `VoiceInput`, `VoiceOutput`,
/// `ConversationBackend`. It has no idea that the backend is Pie, that
/// the input is a microphone rather than a file, or that the output is
/// Apple's synthesizer. Swapping any one of them is a constructor change.
///
/// All published state is mutated on the main queue. The audio layers
/// already marshal their delegate callbacks there; the backend's token
/// deltas are hopped explicitly below.
final class ConversationController: ObservableObject {

    enum State: Equatable {
        case cold
        case idle
        case listening
        case thinking
        case speaking
        case failed(String)

        var isBusy: Bool {
            self == .listening || self == .thinking || self == .speaking
        }
    }

    // MARK: - Published state

    @Published private(set) var state: State = .cold
    @Published private(set) var turns: [Turn] = []
    @Published private(set) var partialTranscript = ""
    @Published private(set) var inputLevel: Float = 0
    @Published private(set) var lastStats: TurnStats?
    @Published private(set) var availability: VoiceInputAvailability?
    @Published private(set) var engineIsWarm = false
    /// Seconds spent in engine boot so far — shown while cold.
    @Published private(set) var bootSeconds = 0
    /// Why the last engine boot failed, if it did. Drives the retry
    /// affordance: a failed *turn* is recoverable by asking again, a
    /// failed *boot* needs the engine started over.
    @Published private(set) var bootFailure: String?

    /// Start listening again as soon as the reply finishes. Off by
    /// default: on a desk with speakers, the synthesizer talks straight
    /// into the microphone and the app answers itself.
    @Published var handsFree = false

    // MARK: - Collaborators

    private let backend: ConversationBackend
    private var input: VoiceInput
    private let output: VoiceOutput
    private let chunker = SentenceChunker()

    /// The next turn should discard whatever conversation state the
    /// backend is holding.
    private var needsFreshSession = true

    /// A backend that falls back to a slower path usually falls back on
    /// every turn. Worth saying once; repeating it down the whole
    /// transcript just reads as breakage.
    private var lastReportedNote = ""

    /// Guards against two concurrent boots.
    private var isBooting = false

    /// Identity of the turn whose completion is still welcome. A barge-in
    /// or "start over" bumps it, so a reply that finishes generating
    /// afterwards is dropped instead of landing on the next question.
    private var activeTurnID = UUID()

    var engineDescription: String { backend.engineDescription }

    init(backend: ConversationBackend, input: VoiceInput, output: VoiceOutput) {
        self.backend = backend
        self.input = input
        self.output = output
        self.input.delegate = self
        self.output.delegate = self
    }

    // MARK: - Lifecycle

    /// Asks for permissions and pays the engine's boot cost up front, so
    /// the first question isn't also the one that waits for a 400 MB
    /// model to load.
    func bootstrap() {
        // Deliberately two tasks. Booting the engine and asking for
        // permissions are independent, and running them in sequence means
        // the model only starts loading once the user has finished
        // dismissing dialogs — several seconds of avoidable waiting.
        bootEngine()
        Task {
            let availability = await input.prepare()
            await MainActor.run { self.availability = availability }
        }
    }

    /// A failed boot leaves half-initialised engine state behind that
    /// cannot be safely re-entered in the same process (the shim refuses
    /// a second boot), so recovery is a clean relaunch: quit now, and the
    /// next tap on the icon starts from scratch.
    func quitToRetryEngineBoot() {
        guard bootFailure != nil else { return }
        print("[app] quitting for a fresh engine boot after: \(bootFailure ?? "")")
        fflush(stdout)
        exit(0)
    }

    private func bootEngine() {
        guard !engineIsWarm, !isBooting else { return }
        isBooting = true
        bootSeconds = 0
        Task {
            let bootStart = Date()
            // Tick the cold status so a hang is visibly different from a
            // slow load — the user (and our screenshots) can see time move.
            let ticker = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                    await MainActor.run {
                        if case .cold = self.state {
                            self.bootSeconds = Int(-bootStart.timeIntervalSinceNow)
                        }
                    }
                }
            }
            let bootError = await backend.warmUp()
            ticker.cancel()
            await MainActor.run {
                self.isBooting = false
                if let bootError {
                    self.bootFailure = bootError
                    self.state = .failed("engine boot failed: \(bootError)")
                } else {
                    self.bootFailure = nil
                    self.engineIsWarm = true
                    if case .cold = self.state { self.state = .idle }
                    if case .failed = self.state { self.state = .idle }
                }
            }
        }
    }

    /// Replaces the input source at runtime — used to switch between the
    /// microphone and a bundled recording.
    func useInput(_ newInput: VoiceInput) {
        input.stop()
        input.delegate = nil
        input = newInput
        input.delegate = self
        Task {
            let availability = await input.prepare()
            await MainActor.run { self.availability = availability }
        }
    }

    // MARK: - Intents

    func toggleListening() {
        switch state {
        case .listening:
            input.stop()
        case .speaking:
            // Barge-in: stop talking and listen instead. The reply may
            // still be generating on the engine queue; retire its turn id
            // so whatever it produces later is discarded.
            activeTurnID = UUID()
            output.cancel()
            chunker.reset()
            if let index = turns.indices.last, turns[index].isStreaming {
                turns[index].isStreaming = false
                if turns[index].text.isEmpty { turns.remove(at: index) }
            }
            beginListening()
        case .idle, .failed:
            beginListening()
        case .cold, .thinking:
            break
        }
    }

    /// Types an utterance instead of speaking it. Goes through exactly
    /// the same path as a transcription, which is what makes the app
    /// usable where speech recognition isn't.
    func submit(typed text: String) {
        let utterance = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !utterance.isEmpty, !state.isBusy, engineIsWarm, bootFailure == nil else { return }
        handle(utterance: utterance)
    }

    func startOver() {
        activeTurnID = UUID()
        input.stop()
        input.resetSequence()
        output.cancel()
        chunker.reset()
        turns = []
        partialTranscript = ""
        lastStats = nil
        needsFreshSession = true
        lastReportedNote = ""
        if let bootFailure {
            state = .failed("engine boot failed: \(bootFailure)")
        } else {
            state = engineIsWarm ? .idle : .cold
        }
    }

    // MARK: - Listening

    private func beginListening() {
        guard engineIsWarm else {
            // A boot that failed can be retried; one still in progress
            // just needs patience. Either way the mic can't help yet.
            state = bootFailure.map { .failed("engine boot failed: \($0)") } ?? .cold
            return
        }
        guard availability?.isReady == true else {
            // Availability is a snapshot from launch. Speech assets finish
            // downloading, the user flips a permission in Settings, the
            // network comes back — re-check before declaring defeat.
            Task {
                let refreshed = await input.prepare()
                await MainActor.run {
                    self.availability = refreshed
                    if refreshed.isReady {
                        self.startInput()
                    } else {
                        self.state = .failed(self.availabilityMessage)
                    }
                }
            }
            return
        }
        startInput()
    }

    private func startInput() {
        partialTranscript = ""
        do {
            try input.start()
            state = .listening
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private var availabilityMessage: String {
        switch availability {
        case .denied(let reason), .unavailable(let reason): return reason
        case .ready, .none: return "Speech input is not ready"
        }
    }

    // MARK: - Answering

    private func handle(utterance: String) {
        partialTranscript = ""
        inputLevel = 0
        turns.append(Turn(speaker: .user, text: utterance))
        turns.append(Turn(speaker: .assistant, text: "", isStreaming: true))
        state = .thinking
        chunker.reset()

        let startingFresh = needsFreshSession
        let turnID = UUID()
        activeTurnID = turnID

        Task {
            do {
                let result = try await backend.reply(
                    to: utterance,
                    startingFresh: startingFresh
                ) { [weak self] delta in
                    // Arrives on the backend's own thread.
                    DispatchQueue.main.async { self?.consume(delta: delta, for: turnID) }
                }
                await MainActor.run {
                    guard self.activeTurnID == turnID else { return }
                    self.finishTurn(with: result.text, stats: result.stats)
                }
            } catch {
                await MainActor.run {
                    guard self.activeTurnID == turnID else { return }
                    self.failTurn(error)
                }
            }
        }
    }

    /// One chunk of generated text: shown immediately, spoken a sentence
    /// at a time so speech overlaps the rest of the generation.
    private func consume(delta: String, for turnID: UUID) {
        guard turnID == activeTurnID else { return }
        guard state == .thinking || state == .speaking else { return }
        guard let index = turns.indices.last, turns[index].isStreaming else { return }
        turns[index].text += delta

        for sentence in chunker.push(delta) {
            output.enqueue(sentence)
        }
    }

    private func finishTurn(with text: String, stats: TurnStats) {
        needsFreshSession = false
        engineIsWarm = true
        bootFailure = nil

        var stats = stats
        if stats.note == lastReportedNote {
            stats.note = ""
        } else {
            lastReportedNote = stats.note
        }
        lastStats = stats

        if let remainder = chunker.flush() {
            output.enqueue(remainder)
        }
        output.finishTurn()

        if let index = turns.indices.last {
            // The inferlet's return value is the authoritative transcript;
            // the streamed text can lag it by a token.
            if !text.isEmpty {
                turns[index].text = text
            }
            // A turn that generated tokens but produced nothing sayable
            // is a real failure mode worth seeing, not a blank bubble.
            if turns[index].text.isEmpty {
                turns[index].text = "(no speakable reply)"
            }
            turns[index].stats = stats
            turns[index].isStreaming = false
        }

        // If nothing is being spoken, the turn is already over.
        if !output.isSpeaking {
            state = .idle
            resumeIfHandsFree()
        } else {
            state = .speaking
        }
    }

    private func failTurn(_ error: Error) {
        chunker.reset()
        output.cancel()
        if let index = turns.indices.last, turns[index].speaker == .assistant {
            turns.remove(at: index)
        }
        // The conversation's KV snapshot may be in an unknown state after
        // a failed turn; start the next one from scratch rather than
        // resuming into whatever was left behind.
        needsFreshSession = true
        state = .failed(error.localizedDescription)
    }

    private func resumeIfHandsFree() {
        guard handsFree, availability?.isReady == true else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
            guard let self, self.state == .idle else { return }
            self.beginListening()
        }
    }
}

// MARK: - VoiceInputDelegate

extension ConversationController: VoiceInputDelegate {

    func voiceInputDidChangeAvailability(_ availability: VoiceInputAvailability) {
        self.availability = availability
    }

    func voiceInputDidUpdatePartial(_ text: String) {
        partialTranscript = text
    }

    func voiceInputDidFinalize(_ text: String) {
        inputLevel = 0
        let utterance = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !utterance.isEmpty else {
            partialTranscript = ""
            if state == .listening { state = .idle }
            return
        }
        handle(utterance: utterance)
    }

    func voiceInputDidUpdateLevel(_ level: Float) {
        inputLevel = level
    }

    func voiceInputDidFail(_ error: Error) {
        inputLevel = 0
        partialTranscript = ""
        state = .failed(error.localizedDescription)
    }
}

// MARK: - VoiceOutputDelegate

extension ConversationController: VoiceOutputDelegate {

    func voiceOutputDidStartSpeaking() {
        if state == .thinking || state == .idle {
            state = .speaking
        }
    }

    func voiceOutputDidFinishSpeaking() {
        guard state == .speaking else { return }
        state = .idle
        resumeIfHandsFree()
    }
}
