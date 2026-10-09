import AVFoundation
import Foundation
import Speech

/// What the microphone does with each buffer, on the input tap's thread:
/// metering, voice-activity detection, the pre-roll, and feeding the
/// recogniser.
///
/// One pipeline serves one microphone input for the life of the app, and
/// its voice-activity detector with it: what the detector has learnt about
/// the reply's echo residue carries from one listening session to the next
/// (voice mode restarts the microphone after a sample question and after
/// mute), and is dropped only when the input format says the route
/// changed.
///
/// It runs on the tap thread so that the moment of onset and the first
/// audio the recogniser receives line up exactly: the pre-roll is handed
/// over and live audio follows with no gap, which a hop through the main
/// thread could not promise. The main thread steers it through the
/// lock-protected methods below and hears from it through `emit`, which
/// is called on the tap thread with the session the event belongs to.
final class CapturePipeline {

    enum Event {
        /// Normalised levels of consecutive windows `window` seconds long.
        case levels([Float], window: TimeInterval)
        /// Voice mode: the user started talking. The request already holds
        /// the pre-roll and keeps receiving live audio.
        case onset(SFSpeechAudioBufferRecognitionRequest)
        /// The request has stopped receiving audio: trailing silence, the
        /// utterance length cap, or the input format changed under it.
        case audioEnded(SFSpeechAudioBufferRecognitionRequest)
    }

    enum Replacement {
        /// The new request took over mid-utterance and keeps receiving
        /// live audio.
        case live
        /// The utterance's audio had already ended; the new request has
        /// all of it and should be told so.
        case ended
        /// Nothing of the utterance was kept to hand over.
        case unavailable
    }

    /// Meter and detector resolution: the 30 Hz the meters want.
    static let window: TimeInterval = 1.0 / 30
    /// Dictation: the most audio held between the end of one utterance and
    /// the start of the next. The gap lasts until the first utterance's
    /// final result, which the final-result grace bounds at a few seconds;
    /// this only bounds the memory if that ever fails.
    static let holdLimit: TimeInterval = 10
    /// Audio kept from before onset, so the first syllable, spoken before
    /// the detector was sure, still reaches the recogniser. Onset over a
    /// reply comes well after its 0.2 s of evidence, because a voice's
    /// quieter syllables fall under the stricter thresholds there and the
    /// evidence leaks away between syllables: in the detector's
    /// simulations, half the time 0.45 s or more after the first syllable,
    /// and one time in ten 1 s or more. Not longer: over a reply, what
    /// comes before the user's first word is the reply's residue, and the
    /// recogniser would transcribe that too.
    static let preRollDuration: TimeInterval = 1.0
    /// Audio kept in case the on-device recogniser turns out to have no
    /// language assets and the utterance has to be recognised again over
    /// the network. Once the recogniser has produced a word it is clearly
    /// working and nothing more is kept.
    static let replayLimit: TimeInterval = 20

    private let lock = NSLock()
    private let playbackEcho: () -> PlaybackEcho
    private let emit: (_ session: Int, _ event: Event) -> Void

    private var session = 0
    private var running = false
    /// Voice mode: nothing reaches the recogniser before onset.
    private var gated = false
    private var onDevice = false
    private var detector = VoiceActivityDetector()
    /// The input format the detector's references were learnt in; a
    /// different one means a different route.
    private var detectorFormat: AVAudioFormat?
    private var format: AVAudioFormat?
    private var preRoll: [AVAudioPCMBuffer] = []
    private var preRollTotal: TimeInterval = 0
    /// The request being fed live audio, if any.
    private var request: SFSpeechAudioBufferRecognitionRequest?
    /// Dictation: audio that arrived while no request was being fed, for
    /// the next utterance's request.
    private var held: [AVAudioPCMBuffer] = []
    private var heldTotal: TimeInterval = 0
    /// The request whose audio `kept` holds. It outlives the end of that
    /// request's audio, because the asset failure can surface after it.
    private var keptFor: SFSpeechAudioBufferRecognitionRequest?
    private var kept: [AVAudioPCMBuffer] = []
    private var keptTotal: TimeInterval = 0

    init(playbackEcho: @escaping () -> PlaybackEcho, emit: @escaping (_ session: Int, _ event: Event) -> Void) {
        self.playbackEcho = playbackEcho
        self.emit = emit
    }

    // MARK: - Steering (main thread)

    /// Voice mode: wait for onset, then feed one request per utterance.
    func startGated(session: Int, onDevice: Bool) {
        lock.lock()
        defer { lock.unlock() }
        reset()
        detector.restart()
        self.session = session
        self.onDevice = onDevice
        gated = true
        running = true
    }

    /// Dictation: feed everything to `request` from the first buffer, and
    /// whatever follows the end of its audio to the next utterance's
    /// request (`resumeStreaming`).
    func startStreaming(session: Int, request: SFSpeechAudioBufferRecognitionRequest, onDevice: Bool) {
        lock.lock()
        defer { lock.unlock() }
        reset()
        self.session = session
        self.onDevice = onDevice
        self.request = request
        keptFor = onDevice ? request : nil
        gated = false
        running = true
    }

    func stop() {
        lock.lock()
        defer { lock.unlock() }
        reset()
    }

    /// Stops feeding `request` if it is still being fed. In voice mode the
    /// detector goes back to waiting for the next onset; in dictation what
    /// follows is held for the next utterance.
    func endAudio(of request: SFSpeechAudioBufferRecognitionRequest) {
        lock.lock()
        defer { lock.unlock() }
        guard self.request === request else { return }
        self.request = nil
        detector.endSpeech()
    }

    /// Dictation: the next utterance's request takes over, fed everything
    /// held since the last one's audio ended and then the live audio, so a
    /// word spoken while the last utterance waited for its final result is
    /// not lost.
    func resumeStreaming(with request: SFSpeechAudioBufferRecognitionRequest, onDevice: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard running, !gated, self.request == nil else { return }
        self.onDevice = onDevice
        dropKept()
        keptFor = onDevice ? request : nil
        for buffer in held {
            request.append(buffer)
            keep(buffer, for: request, copying: false)
        }
        held.removeAll()
        heldTotal = 0
        self.request = request
    }

    /// What the voice-activity detector has learnt, in dBFS.
    struct DetectorReadings {
        let floor: Float
        let echoResidue: Float
        /// The level onset needs while a reply plays.
        let echoOnsetLevel: Float
    }

    /// Read by the audio self-check.
    var detectorReadings: DetectorReadings {
        lock.lock()
        defer { lock.unlock() }
        return DetectorReadings(
            floor: detector.floor,
            echoResidue: detector.echoResidue,
            echoOnsetLevel: detector.echoOnsetLevel
        )
    }

    /// Requests created from now on use (or not) on-device recognition.
    func setOnDevice(_ onDevice: Bool) {
        lock.lock()
        defer { lock.unlock() }
        self.onDevice = onDevice
    }

    /// The recogniser has produced words for `request`, so the audio kept
    /// for a network retry is not needed.
    func stopKeeping(for request: SFSpeechAudioBufferRecognitionRequest) {
        lock.lock()
        defer { lock.unlock() }
        guard keptFor === request else { return }
        dropKept()
    }

    /// Hands the utterance `old` was recognising to `new`: everything kept
    /// of it, then (if it is still going) the live audio.
    func replace(
        _ old: SFSpeechAudioBufferRecognitionRequest,
        with new: SFSpeechAudioBufferRecognitionRequest
    ) -> Replacement {
        lock.lock()
        defer { lock.unlock() }
        let hadAudio = keptFor === old && !kept.isEmpty
        if keptFor === old {
            for buffer in kept {
                new.append(buffer)
            }
            // The replacement is the network recogniser, which needs no
            // second chance.
            dropKept()
        }
        if request === old {
            request = new
            return .live
        }
        return hadAudio ? .ended : .unavailable
    }

    // MARK: - Tap thread

    func process(_ buffer: AVAudioPCMBuffer) {
        let rate = buffer.format.sampleRate
        guard rate > 0 else { return }
        let windowFrames = max(1, Int((rate * Self.window).rounded()))
        let rms = AudioLevel.windowRMS(buffer, windowFrames: windowFrames)
        guard !rms.isEmpty else { return }
        let decibels = rms.map(AudioLevel.decibels(rms:))
        let echo = playbackEcho()
        var events: [Event] = [
            .levels(decibels.map(AudioLevel.normalized(decibels:)), window: Double(windowFrames) / rate),
        ]

        lock.lock()
        guard running else {
            lock.unlock()
            return
        }
        let session = self.session

        if let format, format != buffer.format {
            // The hardware changed under the session (AirPods connected,
            // the engine rebuilt). A recognition request is fed one format
            // only, so the utterance in progress ends here and the
            // pre-roll (or held audio) in the old format is dropped.
            preRoll.removeAll()
            preRollTotal = 0
            held.removeAll()
            heldTotal = 0
            if let current = request {
                request = nil
                detector.endSpeech()
                events.append(.audioEnded(current))
            }
        }
        format = buffer.format

        if !gated {
            if let current = request {
                current.append(buffer)
                keep(buffer, for: current)
            } else if let copy = buffer.copied() {
                hold(copy)
            }
        } else {
            if detectorFormat != buffer.format {
                // What the detector learnt belongs to another route, or to
                // nothing yet.
                if detectorFormat != nil {
                    detector.forgetRoute()
                }
                detectorFormat = buffer.format
            }
            var onset = false
            var ended = false
            for (index, level) in decibels.enumerated() {
                let frames = min(windowFrames, Int(buffer.frameLength) - index * windowFrames)
                switch detector.process(level: level, duration: Double(frames) / rate, echo: echo) {
                case .onset?:
                    onset = true
                case .end?:
                    ended = true
                case nil:
                    break
                }
                // The rest of a buffer that ended an utterance is silence
                // by definition; it must not start the next one.
                if ended { break }
            }

            if let current = request {
                current.append(buffer)
                keep(buffer, for: current)
                if ended {
                    request = nil
                    events.append(.audioEnded(current))
                }
            } else {
                if let copy = buffer.copied() {
                    preRoll.append(copy)
                    preRollTotal += copy.duration
                    while let first = preRoll.first, preRollTotal - first.duration >= Self.preRollDuration {
                        preRoll.removeFirst()
                        preRollTotal -= first.duration
                    }
                }
                if onset {
                    let started = SpeechRecognition.bufferRequest(onDevice: onDevice)
                    dropKept()
                    keptFor = onDevice ? started : nil
                    for piece in preRoll {
                        started.append(piece)
                        keep(piece, for: started, copying: false)
                    }
                    preRoll.removeAll()
                    preRollTotal = 0
                    request = started
                    events.append(.onset(started))
                }
            }
        }
        lock.unlock()

        for event in events {
            emit(session, event)
        }
    }

    // MARK: - Internals (lock held)

    private func reset() {
        running = false
        gated = false
        format = nil
        preRoll.removeAll()
        preRollTotal = 0
        held.removeAll()
        heldTotal = 0
        request = nil
        dropKept()
    }

    private func hold(_ buffer: AVAudioPCMBuffer) {
        held.append(buffer)
        heldTotal += buffer.duration
        while let first = held.first, heldTotal - first.duration >= Self.holdLimit {
            held.removeFirst()
            heldTotal -= first.duration
        }
    }

    private func keep(_ buffer: AVAudioPCMBuffer, for owner: SFSpeechAudioBufferRecognitionRequest, copying: Bool = true) {
        guard keptFor === owner else { return }
        guard keptTotal < Self.replayLimit, let stored = copying ? buffer.copied() : buffer else {
            dropKept()
            return
        }
        kept.append(stored)
        keptTotal += stored.duration
    }

    private func dropKept() {
        keptFor = nil
        kept.removeAll()
        keptTotal = 0
    }
}
