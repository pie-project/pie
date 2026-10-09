import AVFoundation
import Foundation
import Speech

/// `SpeechInput` from the microphone, transcribed by the Speech framework.
///
/// On-device recognition is requested whenever the device supports it, so
/// that in the intended configuration nothing spoken to this app leaves
/// the phone: the audio is transcribed locally and answered by a model
/// that is also local. When on-device recognition turns out to be
/// unusable the class falls back to the network recogniser and says so
/// through `speechInputDidChangeAvailability`, rather than quietly
/// shipping audio to a server.
///
/// Two modes share the engine and the recogniser:
///   - `.dictation`: the raw microphone, streamed to the recogniser from
///     the start. An utterance ends 1.3 s after its last recognised word;
///     the microphone stays open and what is said next goes to the next
///     utterance (it is held while the last one waits for its final
///     result, so a word spoken in that gap is not lost). The session ends
///     on `stop()`, after an utterance in which nothing was recognised
///     (8 s of nothing), or on an error.
///   - `.conversation`: the echo-cancelled microphone, open until
///     `cancel()`. Nothing is transcribed until the voice-activity
///     detector hears the user start (`speechInputDidDetectSpeechOnset`);
///     each utterance then gets its own recognition task, fed the pre-roll
///     and the live audio, and ends on trailing silence, after which the
///     microphone waits for the next one. Because the reply is played
///     through the same engine and cancelled out of the input, this runs
///     while the reply plays: that is how the user talks over it. One
///     utterance the recogniser fails on is an empty utterance, not the end
///     of the session.
///
/// Call from the main thread. Delegate calls arrive on the main queue.
final class MicrophoneInput: NSObject, SpeechInput {

    weak var delegate: SpeechInputDelegate?
    private(set) var isListening = false

    /// Dictation: silence after the last recognised word that ends the
    /// utterance. Long enough to think mid-sentence, short enough that the
    /// app doesn't feel deaf.
    private let dictationSilence: TimeInterval = 1.3
    /// Dictation: on-device recognition never errors on silence, so
    /// without this a press that hears nothing would listen until the
    /// one-minute system limit.
    private let noSpeechTimeout: TimeInterval = 8.0
    /// The final result usually follows the end of audio within a few
    /// hundred milliseconds. If it doesn't, the last partial is what was
    /// said.
    private let finalResultGrace: TimeInterval = 1.0
    /// The same, for audio replayed to the network recogniser after the
    /// on-device one failed.
    private let networkReplayGrace: TimeInterval = 4.0
    /// Voice mode: utterances in a row lost to recogniser errors before the
    /// session gives up and reports the error. One is a glitch; several
    /// mean the recogniser is not working, and an always-on microphone
    /// that silently hears nothing is worse than a visible failure.
    private let failuresBeforeGivingUp = 3

    private let recognizer = SpeechRecognition.makeRecognizer()
    private let hub = AudioEngineHub.shared
    private lazy var pipeline = CapturePipeline(
        echoLikely: { [hub] in hub.isEchoLikely },
        emit: { [weak self] session, event in
            DispatchQueue.main.async {
                self?.handle(event, session: session)
            }
        }
    )

    private var mode: ListeningMode = .dictation
    /// Bumped by every start and every end of a session. Deferred work
    /// (pipeline events, spaced level reports, the final-result grace)
    /// carries the session it belongs to and is dropped once it is over,
    /// so nothing from a cancelled session reaches the delegate.
    private var session = 0
    private var utterance: Utterance?
    private var dictationTimer: Timer?
    private var usesOnDeviceRecognition = false
    /// What the delegate was last told about on-device recognition. The
    /// sample question can find on-device recognition unusable and switch
    /// the whole app to the network recogniser; the microphone says so the
    /// next time it starts, rather than shipping audio to a server while
    /// the UI still promises it stays on the phone.
    private var reportedOnDevice: Bool?
    private var permissionsGranted = false
    /// Dictation: the utterance in progress is the session's last, because
    /// `stop()` was called.
    private var closing = false
    /// Voice mode: utterances in a row the recogniser failed on.
    private var consecutiveFailures = 0

    /// One recognised utterance.
    private final class Utterance {
        var request: SFSpeechAudioBufferRecognitionRequest
        var task: SFSpeechRecognitionTask?
        var onDevice: Bool
        var startedAt = Date()
        var transcript = ""
        /// The recogniser has produced at least one word.
        var heardWords = false
        /// No more audio is coming: trailing silence, `stop()`, or the
        /// end of the session.
        var audioEnded = false
        var finalized = false

        init(request: SFSpeechAudioBufferRecognitionRequest, onDevice: Bool) {
            self.request = request
            self.onDevice = onDevice
        }
    }

    override init() {
        super.init()
        recognizer?.delegate = self
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(audioWasInterrupted(_:)),
            name: AudioEngineHub.audioWasInterrupted,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(inputWasLost(_:)),
            name: AudioEngineHub.inputWasLost,
            object: nil
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - SpeechInput

    func prepare() async -> VoiceInputAvailability {
        guard let recognizer else {
            return .unavailable("No English speech recogniser on this device. Type instead.")
        }
        guard await SpeechPermissions.requestRecognition() else {
            return .denied("Speech recognition permission was declined. Type instead, or allow it in Settings.")
        }
        guard await SpeechPermissions.requestMicrophone() else {
            return .denied("Microphone permission was declined. Type instead, or allow it in Settings.")
        }
        return await MainActor.run {
            self.permissionsGranted = true
            guard recognizer.isAvailable else {
                return .unavailable(SpeechRecognition.unavailableMessage)
            }
            self.usesOnDeviceRecognition = OnDeviceRecognition.isAvailable(on: recognizer)
            self.reportedOnDevice = self.usesOnDeviceRecognition
            return .ready(onDevice: self.usesOnDeviceRecognition)
        }
    }

    func start(mode: ListeningMode) throws {
        if isListening {
            if mode == self.mode { return }
            cancel()
        }
        guard let recognizer, recognizer.isAvailable else {
            throw VoiceInputError.recogniserUnavailable
        }

        session += 1
        let session = self.session
        self.mode = mode
        closing = false
        consecutiveFailures = 0
        usesOnDeviceRecognition = OnDeviceRecognition.isAvailable(on: recognizer)

        switch mode {
        case .conversation:
            pipeline.startGated(session: session, onDevice: usesOnDeviceRecognition)
        case .dictation:
            let request = SpeechRecognition.bufferRequest(onDevice: usesOnDeviceRecognition)
            pipeline.startStreaming(session: session, request: request, onDevice: usesOnDeviceRecognition)
            begin(Utterance(request: request, onDevice: usesOnDeviceRecognition))
        }

        do {
            let pipeline = self.pipeline
            try hub.startInput(mode == .conversation ? .voiceProcessed : .plain) { buffer, _ in
                pipeline.process(buffer)
            }
        } catch {
            pipeline.stop()
            discardUtterance()
            self.session += 1
            throw error
        }

        isListening = true
        if mode == .dictation {
            armDictationTimer(after: noSpeechTimeout)
        }
        reportOnDeviceIfChanged()
    }

    /// Dictation: ends the utterance and the session; the transcript is
    /// finalised once the recogniser has had its last word. Voice mode:
    /// ends the utterance in progress, if any, and keeps listening.
    func stop() {
        guard isListening else { return }
        switch mode {
        case .conversation:
            if let current = utterance, !current.audioEnded {
                endAudio(of: current)
            }
        case .dictation:
            closing = true
            dictationTimer?.invalidate()
            dictationTimer = nil
            // The microphone closes now, as the user expects when they tap
            // stop; recognition finishes on the audio already captured.
            hub.stopInput()
            if let current = utterance {
                endAudio(of: current)
            } else {
                let target = delegate
                endSession()
                target?.speechInputDidFinalize("")
            }
        }
    }

    func cancel() {
        guard isListening else { return }
        endSession()
    }

    // MARK: - Pipeline events

    private func handle(_ event: CapturePipeline.Event, session: Int) {
        guard session == self.session, isListening else { return }
        switch event {
        case .levels(let levels, let window):
            deliverLevels(levels, spacedBy: window, session: session)

        case .onset(let request):
            // The previous utterance may still be waiting for its final
            // result; the user has moved on, so its best transcript stands.
            if let previous = utterance {
                finalize(previous)
            }
            begin(Utterance(request: request, onDevice: request.requiresOnDeviceRecognition))
            delegate?.speechInputDidDetectSpeechOnset()

        case .audioEnded(let request):
            guard let current = utterance, current.request === request else { return }
            endAudio(of: current)
        }
    }

    /// Spreads one tap buffer's worth of levels (the tap delivers about
    /// 100 ms at a time) over the time they cover, so the meter moves at
    /// 30 Hz instead of jumping ten times a second.
    private func deliverLevels(_ levels: [Float], spacedBy window: TimeInterval, session: Int) {
        for (index, level) in levels.enumerated() {
            if index == 0 {
                delegate?.speechInputDidUpdateLevel(level)
                continue
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + window * Double(index)) { [weak self] in
                guard let self, self.session == session else { return }
                self.delegate?.speechInputDidUpdateLevel(level)
            }
        }
    }

    // MARK: - Recognition

    private func begin(_ current: Utterance) {
        utterance = current
        startTask(for: current)
    }

    private func startTask(for current: Utterance) {
        guard let recognizer else { return }
        let request = current.request
        current.startedAt = Date()
        // Results arrive on the recogniser's queue, which is the main
        // queue.
        current.task = recognizer.recognitionTask(with: request) { [weak self, weak current] result, error in
            // A replaced or finished request can still deliver one last
            // callback; only the utterance's live request speaks for it.
            guard let self, let current, !current.finalized, current.request === request else { return }
            self.recognition(of: current, result: result, error: error)
        }
    }

    private func recognition(of current: Utterance, result: SFSpeechRecognitionResult?, error: Error?) {
        if let result {
            let text = result.bestTranscription.formattedString
            if !text.isEmpty {
                if !current.heardWords {
                    current.heardWords = true
                    consecutiveFailures = 0
                    pipeline.stopKeeping(for: current.request)
                    if mode == .dictation {
                        delegate?.speechInputDidDetectSpeechOnset()
                    }
                }
                current.transcript = text
                delegate?.speechInputDidUpdatePartial(text)
                if mode == .dictation, !current.audioEnded {
                    armDictationTimer(after: dictationSilence)
                }
            }
            if result.isFinal {
                finalize(current)
                return
            }
        }
        if let error {
            recognitionFailed(current, error: error)
        }
    }

    private func recognitionFailed(_ current: Utterance, error: Error) {
        // A recogniser that gives up after the user has already said
        // something is not a failure: it is the end of the utterance, and
        // the transcript stands. Likewise when the audio was ended on
        // purpose: what was heard, possibly nothing, is the utterance.
        if !current.transcript.isEmpty || current.audioEnded {
            finalize(current)
            return
        }
        if current.onDevice, -current.startedAt.timeIntervalSinceNow < OnDeviceRecognition.assetFailureWindow {
            fallBackToNetwork(current)
            return
        }
        if SpeechRecognition.isNoSpeech(error) {
            finalize(current)
            return
        }
        if mode == .conversation, recognizer?.isAvailable == true {
            // The session outlives one utterance the recogniser could not
            // handle (a transient recogniser or server error): that
            // utterance is empty and the microphone waits for the next.
            consecutiveFailures += 1
            if consecutiveFailures < failuresBeforeGivingUp {
                let code = (error as NSError).code
                print("[audio] recognition failed for one utterance (\((error as NSError).domain) \(code)); still listening")
                finalize(current)
                return
            }
        }
        let target = delegate
        endSession()
        target?.speechInputDidFail(error)
    }

    /// The local recogniser has no usable assets. The utterance's audio
    /// was kept, so it is handed to the network recogniser and the user
    /// does not have to repeat themselves.
    private func fallBackToNetwork(_ current: Utterance) {
        OnDeviceRecognition.markUnusable()
        usesOnDeviceRecognition = false
        pipeline.setOnDevice(false)

        let replacement = SpeechRecognition.bufferRequest(onDevice: false)
        let old = current.request
        current.request = replacement
        current.onDevice = false
        // Swapped before the old request is ended, so the tap never
        // appends to a request that has been told its audio is over.
        let handover = pipeline.replace(old, with: replacement)
        old.endAudio()
        current.task?.cancel()
        startTask(for: current)
        reportedOnDevice = false
        if handover != .live {
            // The utterance's audio was already over: the network
            // recogniser gets all of it at once, plus the time a round
            // trip takes, before the best partial stands.
            replacement.endAudio()
            current.audioEnded = true
            armFinalizeGrace(for: current, after: networkReplayGrace)
        }
        delegate?.speechInputDidChangeAvailability(.ready(onDevice: false))
    }

    private func endAudio(of current: Utterance) {
        guard !current.audioEnded else { return }
        current.audioEnded = true
        pipeline.endAudio(of: current.request)
        current.request.endAudio()
        armFinalizeGrace(for: current, after: finalResultGrace)
    }

    private func armFinalizeGrace(for current: Utterance, after interval: TimeInterval) {
        let session = self.session
        DispatchQueue.main.asyncAfter(deadline: .now() + interval) { [weak self, weak current] in
            guard let self, let current, self.session == session else { return }
            self.finalize(current)
        }
    }

    private func finalize(_ current: Utterance) {
        guard !current.finalized else { return }
        current.finalized = true
        if !current.audioEnded {
            current.audioEnded = true
            pipeline.endAudio(of: current.request)
            current.request.endAudio()
        }
        pipeline.stopKeeping(for: current.request)
        current.task?.cancel()
        current.task = nil
        if utterance === current {
            utterance = nil
        }

        let transcript = current.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        let target = delegate
        guard mode == .dictation else {
            target?.speechInputDidFinalize(transcript)
            return
        }
        // Dictation goes on to the next utterance after a pause that
        // followed words. After `stop()`, or an utterance that heard
        // nothing, the session is over.
        guard isListening, !closing, !transcript.isEmpty else {
            endSession()
            target?.speechInputDidFinalize(transcript)
            return
        }
        let session = self.session
        target?.speechInputDidFinalize(transcript)
        // The delegate may have stopped, cancelled or restarted the session
        // in answer.
        guard session == self.session, isListening, !closing, utterance == nil else { return }
        let next = SpeechRecognition.bufferRequest(onDevice: usesOnDeviceRecognition)
        pipeline.resumeStreaming(with: next, onDevice: usesOnDeviceRecognition)
        begin(Utterance(request: next, onDevice: usesOnDeviceRecognition))
        armDictationTimer(after: noSpeechTimeout)
    }

    /// Tells the delegate when on-device recognition was given up since it
    /// last heard, before this session's audio reaches a recogniser that
    /// is not on the phone.
    private func reportOnDeviceIfChanged() {
        guard reportedOnDevice != usesOnDeviceRecognition else { return }
        reportedOnDevice = usesOnDeviceRecognition
        delegate?.speechInputDidChangeAvailability(.ready(onDevice: usesOnDeviceRecognition))
    }

    // MARK: - Session

    private func endSession() {
        session += 1
        pipeline.stop()
        discardUtterance()
        dictationTimer?.invalidate()
        dictationTimer = nil
        hub.stopInput()
        isListening = false
    }

    private func discardUtterance() {
        guard let current = utterance else { return }
        utterance = nil
        current.finalized = true
        current.task?.cancel()
        current.task = nil
    }

    private func armDictationTimer(after interval: TimeInterval) {
        dictationTimer?.invalidate()
        let session = self.session
        let timer = Timer(timeInterval: interval, repeats: false) { [weak self] _ in
            guard let self, self.session == session else { return }
            self.dictationTimerFired()
        }
        RunLoop.main.add(timer, forMode: .common)
        dictationTimer = timer
    }

    private func dictationTimerFired() {
        dictationTimer = nil
        guard isListening, mode == .dictation else { return }
        guard let current = utterance, current.heardWords else {
            // Nothing recognised since the microphone opened or since the
            // last utterance: the user is not dictating any more.
            stop()
            return
        }
        // A pause after words ends this utterance only. The microphone
        // stays open; what is said next is held for the next utterance,
        // which starts once this one's final result is in.
        endAudio(of: current)
    }

    /// Siri, a call, or an alarm took the audio. Dictation keeps what was
    /// heard and finalises it, as before. Voice mode ends: answering a
    /// half-heard question in the middle of a phone call would be worse
    /// than asking the user to tap to resume.
    @objc private func audioWasInterrupted(_ note: Notification) {
        audioWentAway(VoiceInputError.interrupted)
    }

    /// The microphone's input route went away during a hardware change and
    /// did not come back (playback, if any, carries on).
    @objc private func inputWasLost(_ note: Notification) {
        audioWentAway(VoiceInputError.inputLost)
    }

    private func audioWentAway(_ error: VoiceInputError) {
        guard isListening else { return }
        switch mode {
        case .dictation:
            stop()
        case .conversation:
            let target = delegate
            endSession()
            target?.speechInputDidFail(error)
        }
    }
}

// MARK: - SFSpeechRecognizerDelegate

extension MicrophoneInput: SFSpeechRecognizerDelegate {

    /// Availability flips at runtime: assets finish downloading, the
    /// network drops, Siri takes the recogniser. Without this the app
    /// would keep whatever it saw at launch until the next relaunch.
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        guard permissionsGranted else { return }
        let availability: VoiceInputAvailability
        if available {
            usesOnDeviceRecognition = OnDeviceRecognition.isAvailable(on: speechRecognizer)
            reportedOnDevice = usesOnDeviceRecognition
            availability = .ready(onDevice: usesOnDeviceRecognition)
        } else {
            availability = .unavailable(SpeechRecognition.unavailableMessage)
        }
        delegate?.speechInputDidChangeAvailability(availability)
    }
}
