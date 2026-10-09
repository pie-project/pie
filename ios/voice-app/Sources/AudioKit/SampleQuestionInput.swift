import AVFoundation
import Foundation
import Speech

/// Bundled recordings played as if spoken: each `start` plays the next one
/// aloud and finalises its transcript only once playback has ended.
///
/// It drives the voice path with no human in the loop: the same
/// recogniser, the same delegate callbacks, the same controller and model;
/// only the source of the audio differs. The recordings are played in
/// order and wrap around, and the later ones are follow-ups ("say that
/// more simply") that only make sense if the conversation carried over,
/// which makes this a test of the backend's session state as well.
///
/// The recognition request reads the file faster than real time, so its
/// final transcript is ready long before the recording has finished
/// playing. Finalising on that alone is what let the reply start talking
/// over the question; finalising needs both the final transcript and the
/// end of playback.
///
/// The recording plays through the shared engine's cue channel, so when
/// voice mode's echo-cancelled microphone is open it does not hear the
/// question as the user barging in.
///
/// Call from the main thread. Delegate calls arrive on the main queue.
final class SampleQuestionInput: NSObject, SpeechInput {

    weak var delegate: SpeechInputDelegate?
    private(set) var isListening = false

    /// At least one of the recordings is in the bundle.
    var hasRecordings: Bool {
        resources.contains { url(for: $0) != nil }
    }

    /// When the current or last recording stopped playing, naturally or
    /// cut off by `stop()`. Read by the audio self-check.
    private(set) var playbackEndedAt: Date?

    /// If the recogniser has not delivered its final result this long
    /// after playback ended, the best partial is the transcript.
    private let finalResultGrace: TimeInterval = 2.0
    /// Queued in pieces this long; see `PlaybackChannel.attach`.
    private let pieceDuration: TimeInterval = 0.25

    private let resources: [String]
    private let fileExtension: String
    private var index = 0
    private let recognizer = SpeechRecognition.makeRecognizer()
    private let hub = AudioEngineHub.shared
    private var usesOnDeviceRecognition = false

    /// Bumped by every start and every end, so deferred callbacks from a
    /// finished or cancelled question are dropped.
    private var session = 0
    /// Bumped by every recognition attempt, so the dying callbacks of an
    /// on-device attempt do not speak for its network replacement.
    private var attempt = 0
    private var task: SFSpeechRecognitionTask?
    private var transcript = ""
    private var recognitionDone = false
    private var playbackDone = false
    private var meter: Timer?
    private var meterLevel: Float = 0

    init(resources: [String], fileExtension: String = "wav") {
        self.resources = resources
        self.fileExtension = fileExtension
        super.init()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(audioWasInterrupted(_:)),
            name: AudioEngineHub.audioWasInterrupted,
            object: nil
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - SpeechInput

    func prepare() async -> VoiceInputAvailability {
        guard let recognizer else {
            return .unavailable("No English speech recogniser on this device.")
        }
        guard hasRecordings else {
            return .unavailable("No bundled sample recordings were found.")
        }
        guard await SpeechPermissions.requestRecognition() else {
            return .denied("Speech recognition permission was declined.")
        }
        return await MainActor.run {
            guard recognizer.isAvailable else {
                return .unavailable(SpeechRecognition.unavailableMessage)
            }
            self.usesOnDeviceRecognition = OnDeviceRecognition.isAvailable(on: recognizer)
            return .ready(onDevice: self.usesOnDeviceRecognition)
        }
    }

    /// Plays the next recording. The mode is ignored: a recording is one
    /// utterance either way.
    func start(mode: ListeningMode) throws {
        guard !isListening else { return }
        guard let recognizer, recognizer.isAvailable else {
            throw VoiceInputError.recogniserUnavailable
        }
        guard let url = nextRecording() else {
            throw VoiceInputError.missingAudioFile("\(resources.first ?? "sample").\(fileExtension)")
        }
        let pieces = try AVAudioFile.loadPieces(
            of: url,
            as: AudioEngineHub.playbackFormat,
            pieceDuration: pieceDuration
        )
        try hub.beginPlayback(.cue)

        session += 1
        let session = self.session
        isListening = true
        transcript = ""
        recognitionDone = false
        playbackDone = false
        playbackEndedAt = nil
        usesOnDeviceRecognition = OnDeviceRecognition.isAvailable(on: recognizer)

        for (offset, piece) in pieces.enumerated() {
            let isLast = offset == pieces.count - 1
            hub.cue.schedule(piece) { [weak self] in
                guard isLast else { return }
                self?.playbackDidEnd(session: session)
            }
        }
        startMeter(session: session)
        recognize(url, session: session)

        DispatchQueue.main.async { [weak self] in
            guard let self, self.session == session else { return }
            // Playback has started: to the listener, the user is talking.
            self.delegate?.speechInputDidDetectSpeechOnset()
            if pieces.isEmpty {
                self.playbackDidEnd(session: session)
            }
        }
    }

    /// Cuts the recording off and finalises what was recognised.
    func stop() {
        guard isListening else { return }
        if !playbackDone {
            playbackDidEnd(session: session)
        }
    }

    func cancel() {
        guard isListening else { return }
        session += 1
        task?.cancel()
        task = nil
        stopMeter()
        hub.cue.stop()
        if !playbackDone {
            hub.endPlayback(.cue)
        }
        isListening = false
    }

    /// Next `start` plays the first recording again.
    func resetSequence() {
        index = 0
    }

    // MARK: - Playback

    private func url(for resource: String) -> URL? {
        Bundle.main.url(forResource: resource, withExtension: fileExtension)
    }

    /// The recording at the current position, skipping any that are not
    /// in the bundle.
    private func nextRecording() -> URL? {
        guard !resources.isEmpty else { return nil }
        for step in 0..<resources.count {
            let position = (index + step) % resources.count
            if let url = url(for: resources[position]) {
                index = position
                return url
            }
        }
        return nil
    }

    private func playbackDidEnd(session: Int) {
        guard session == self.session, isListening, !playbackDone else { return }
        playbackDone = true
        playbackEndedAt = Date()
        stopMeter()
        delegate?.speechInputDidUpdateLevel(0)
        // Stopping also resets the player's timeline for the next question.
        hub.cue.stop()
        hub.endPlayback(.cue)

        if !recognitionDone {
            DispatchQueue.main.asyncAfter(deadline: .now() + finalResultGrace) { [weak self] in
                guard let self, self.session == session, !self.recognitionDone else { return }
                self.recognitionDone = true
                self.task?.cancel()
                self.finishIfReady()
            }
        }
        finishIfReady()
    }

    private func startMeter(session: Int) {
        stopMeter()
        let timer = Timer(timeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            guard let self, self.session == session else { return }
            let level = self.hub.cue.level
            self.meterLevel = level >= self.meterLevel ? level : self.meterLevel * 0.6 + level * 0.4
            self.delegate?.speechInputDidUpdateLevel(self.meterLevel)
        }
        RunLoop.main.add(timer, forMode: .common)
        meter = timer
    }

    private func stopMeter() {
        meter?.invalidate()
        meter = nil
        meterLevel = 0
    }

    // MARK: - Recognition

    private func recognize(_ url: URL, session: Int) {
        guard let recognizer else { return }
        attempt += 1
        let attempt = self.attempt
        let onDevice = usesOnDeviceRecognition
        let startedAt = Date()
        let request = SFSpeechURLRecognitionRequest(url: url)
        SpeechRecognition.configure(request, onDevice: onDevice)

        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self, self.session == session, self.attempt == attempt, !self.recognitionDone else { return }

            if let result {
                let text = result.bestTranscription.formattedString
                if !text.isEmpty {
                    self.transcript = text
                    self.delegate?.speechInputDidUpdatePartial(text)
                }
                if result.isFinal {
                    self.recognitionDone = true
                    self.finishIfReady()
                    return
                }
            }

            guard let error else { return }
            if !self.transcript.isEmpty {
                self.recognitionDone = true
                self.finishIfReady()
            } else if onDevice, -startedAt.timeIntervalSinceNow < OnDeviceRecognition.assetFailureWindow {
                // A local recogniser that gives up at once, having produced
                // nothing from a recording of a clear question, is missing
                // its language assets. The audio is a file, so it is simply
                // recognised again over the network while it keeps playing.
                // This switches the microphone too, which tells its own
                // listener the next time it starts.
                OnDeviceRecognition.markUnusable()
                self.usesOnDeviceRecognition = false
                self.delegate?.speechInputDidChangeAvailability(.ready(onDevice: false))
                self.recognize(url, session: session)
            } else {
                let target = self.delegate
                self.cancel()
                target?.speechInputDidFail(error)
            }
        }
    }

    private func finishIfReady() {
        guard isListening, recognitionDone, playbackDone else { return }
        isListening = false
        session += 1
        task = nil
        index += 1
        let text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        delegate?.speechInputDidFinalize(text)
    }

    @objc private func audioWasInterrupted(_ note: Notification) {
        guard isListening else { return }
        let target = delegate
        cancel()
        target?.speechInputDidFail(VoiceInputError.interrupted)
    }
}
