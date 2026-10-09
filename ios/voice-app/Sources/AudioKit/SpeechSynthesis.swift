import AVFoundation
import Foundation

/// `SpeechOutput` rendered through the app's shared audio engine.
///
/// `AVSpeechSynthesizer.speak` plays through an audio path of its own that
/// the echo canceller never sees, so the microphone heard the reply as if
/// someone were talking and voice mode could not listen while it spoke.
/// Here the synthesizer only renders (`write(_:toBufferCallback:)`), one
/// sentence at a time and in order, and the buffers play on the hub's
/// speech channel: the same engine whose input node cancels them out.
///
/// Call from the main thread. Delegate calls arrive on the main queue,
/// asynchronously and in order, addressed to the delegate that was set
/// when the event happened: a turn cut off by `stop()` reports its end to
/// whoever owned it, even if a new owner has already taken over.
final class SpeechSynthesis: NSObject, SpeechOutput {

    weak var delegate: SpeechOutputDelegate?
    /// True from the first `enqueue` of a turn until every rendered buffer
    /// has been heard and `finishTurn()` was called, or until `stop()`.
    private(set) var isSpeaking = false
    var voiceIdentifier: String?
    var rate: Float = 0.5
    /// About -14 dB: still audible, so a false alarm costs nothing, but
    /// well under a voice talking over it.
    var isDucked = false {
        didSet { hub.speech.gain = isDucked ? Self.duckedGain : 1 }
    }
    private static let duckedGain: Float = 0.2

    private let synthesizer = AVSpeechSynthesizer()
    private let hub = AudioEngineHub.shared

    /// Names the current turn. Every deferred callback (rendered audio,
    /// played-back completions, the meter) carries the turn it belongs to
    /// and is dropped once that turn is over, so audio from a stopped reply
    /// can never play later.
    private var turn = 0
    private var pending: [String] = []
    /// The sentence the synthesizer is rendering now. One at a time, so
    /// the buffers arrive in speaking order.
    private var rendering: AVSpeechUtterance?
    private var renderingProducedAudio = false
    private var unplayedBuffers = 0
    private var turnIsOpen = false
    private var turnVoice: AVSpeechSynthesisVoice?
    private var meter: Timer?
    private var meterLevel: Float = 0

    override init() {
        super.init()
        synthesizer.delegate = self
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

    // MARK: - SpeechOutput

    func enqueue(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if !isSpeaking {
            guard beginTurn() else { return }
        }
        // Text arriving after `finishTurn()` but before the queue drained
        // continues the same turn, so there is one start and one finish.
        turnIsOpen = true
        pending.append(trimmed)
        renderNextIfIdle()
    }

    func finishTurn() {
        turnIsOpen = false
        // If everything has already been heard, the last completion has
        // been and gone; settle here rather than wait for one that will
        // never come.
        settleIfDone()
    }

    func stop() {
        guard isSpeaking else { return }
        endTurn(interrupted: true)
    }

    /// Installed voices in the current language, best quality first.
    static func availableVoices() -> [SpeechVoiceInfo] {
        rankedVoices().map { voice in
            SpeechVoiceInfo(
                id: voice.identifier,
                name: voice.name,
                quality: qualityName(voice.quality),
                language: voice.language
            )
        }
    }

    // MARK: - Turn

    private func beginTurn() -> Bool {
        turn += 1
        isDucked = false
        let target = delegate
        do {
            try hub.beginPlayback(.speech)
        } catch {
            // Reported as a turn that was cut off at once, so whoever is
            // waiting for the end of speech moves on. Nothing is latched:
            // the next chunk tries the engine again, so a failure while a
            // call holds the audio does not silence the app afterwards.
            print("[audio] speech could not start: \(error.localizedDescription)")
            deliver {
                target?.speechOutputDidStart()
                target?.speechOutputDidFinish(interrupted: true)
            }
            return false
        }
        isSpeaking = true
        turnVoice = resolveVoice()
        startMeter()
        deliver { target?.speechOutputDidStart() }
        return true
    }

    private func settleIfDone() {
        guard isSpeaking, !turnIsOpen, rendering == nil, pending.isEmpty, unplayedBuffers == 0 else { return }
        endTurn(interrupted: false)
    }

    private func endTurn(interrupted: Bool) {
        turn += 1
        if rendering != nil {
            synthesizer.stopSpeaking(at: .immediate)
        }
        hub.speech.stop()
        pending.removeAll()
        rendering = nil
        unplayedBuffers = 0
        turnIsOpen = false
        isSpeaking = false
        meter?.invalidate()
        meter = nil
        meterLevel = 0

        let target = delegate
        deliver {
            target?.speechOutputLevel(0)
            target?.speechOutputDidFinish(interrupted: interrupted)
        }
        hub.endPlayback(.speech)
    }

    // MARK: - Rendering

    private func renderNextIfIdle() {
        guard rendering == nil, !pending.isEmpty else { return }
        let utterance = AVSpeechUtterance(string: pending.removeFirst())
        utterance.voice = turnVoice
        utterance.rate = min(max(rate, AVSpeechUtteranceMinimumSpeechRate), AVSpeechUtteranceMaximumSpeechRate)
        rendering = utterance
        renderingProducedAudio = false

        let turn = self.turn
        // One converter per sentence: it carries resampler state from one
        // buffer to the next, and its tail is flushed when the sentence
        // ends. The coalescer turns the synthesizer's 11 ms pieces into
        // 100 ms buffers before they cost a hop to the main queue each.
        let converter = PCMConverter(outputFormat: AudioEngineHub.playbackFormat)
        let coalescer = BufferCoalescer(format: AudioEngineHub.playbackFormat, chunkFrames: 4800)
        synthesizer.write(utterance) { [weak self] buffer in
            guard let pcm = buffer as? AVAudioPCMBuffer else { return }
            if pcm.frameLength == 0 {
                // The zero-length buffer marks the end of the sentence. It
                // comes through this same callback after the sentence's
                // audio, so it is handled after that audio is queued. The
                // synthesizer has been seen to send it twice; the second
                // finds nothing left to flush and a sentence already over.
                let tail = coalescer.append(converter.finish()) + coalescer.flush()
                DispatchQueue.main.async {
                    self?.play(tail, turn: turn)
                    self?.renderingDidEnd(utterance, turn: turn)
                }
            } else {
                let ready = coalescer.append(converter.convert(pcm))
                guard !ready.isEmpty else { return }
                DispatchQueue.main.async {
                    self?.play(ready, turn: turn)
                }
            }
        }
        // A sentence the synthesizer renders nothing for (a voice that
        // failed to load) must not leave the turn speaking forever.
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
            guard let self, self.rendering === utterance, !self.renderingProducedAudio else { return }
            print("[audio] the synthesizer rendered nothing for a sentence; skipping it")
            self.renderingDidEnd(utterance, turn: turn)
        }
    }

    private func play(_ buffers: [AVAudioPCMBuffer], turn: Int) {
        guard turn == self.turn, isSpeaking else { return }
        if !buffers.isEmpty {
            renderingProducedAudio = true
        }
        for buffer in buffers {
            unplayedBuffers += 1
            hub.speech.schedule(buffer) { [weak self] in
                self?.bufferWasHeard(turn: turn)
            }
        }
    }

    private func bufferWasHeard(turn: Int) {
        guard turn == self.turn else { return }
        unplayedBuffers = max(0, unplayedBuffers - 1)
        settleIfDone()
    }

    private func renderingDidEnd(_ utterance: AVSpeechUtterance, turn: Int) {
        guard turn == self.turn, rendering === utterance else { return }
        rendering = nil
        renderNextIfIdle()
        settleIfDone()
    }

    // MARK: - Meter

    private func startMeter() {
        meter?.invalidate()
        meterLevel = 0
        let turn = self.turn
        let timer = Timer(timeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            guard let self, self.turn == turn else { return }
            let level = self.hub.speech.level
            // Rise at once, fall over a few frames, the way a VU needle
            // does; the raw windows flicker between syllables.
            self.meterLevel = level >= self.meterLevel ? level : self.meterLevel * 0.6 + level * 0.4
            self.delegate?.speechOutputLevel(self.meterLevel)
        }
        // Common modes, so the meter keeps moving while a list scrolls.
        RunLoop.main.add(timer, forMode: .common)
        meter = timer
    }

    // MARK: - Voices

    private func resolveVoice() -> AVSpeechSynthesisVoice? {
        if let voiceIdentifier, let voice = AVSpeechSynthesisVoice(identifier: voiceIdentifier) {
            return voice
        }
        return Self.rankedVoices().first
            ?? AVSpeechSynthesisVoice(language: AVSpeechSynthesisVoice.currentLanguageCode())
    }

    /// Premium, then enhanced, then default voices in the current language.
    /// Novelty voices are not offered, and neither is Personal Voice,
    /// which renders nothing without its own authorisation.
    private static func rankedVoices() -> [AVSpeechSynthesisVoice] {
        let language = AVSpeechSynthesisVoice.currentLanguageCode()
        return AVSpeechSynthesisVoice.speechVoices()
            .filter { voice in
                voice.language == language
                    && !voice.voiceTraits.contains(.isNoveltyVoice)
                    && !voice.voiceTraits.contains(.isPersonalVoice)
            }
            .sorted { lhs, rhs in
                let left = rank(lhs.quality)
                let right = rank(rhs.quality)
                return left != right ? left > right : lhs.name < rhs.name
            }
    }

    private static func rank(_ quality: AVSpeechSynthesisVoiceQuality) -> Int {
        switch quality {
        case .premium: return 3
        case .enhanced: return 2
        default: return 1
        }
    }

    private static func qualityName(_ quality: AVSpeechSynthesisVoiceQuality) -> String {
        switch quality {
        case .premium: return "Premium"
        case .enhanced: return "Enhanced"
        default: return "Default"
        }
    }

    // MARK: - Plumbing

    private func deliver(_ work: @escaping () -> Void) {
        DispatchQueue.main.async(execute: work)
    }

    @objc private func audioWasInterrupted(_ note: Notification) {
        guard isSpeaking else { return }
        endTurn(interrupted: true)
    }
}

extension SpeechSynthesis: AVSpeechSynthesizerDelegate {

    /// The zero-length buffer is what normally ends a sentence. Should it
    /// ever not arrive, the queue must not stall on that sentence, so the
    /// synthesizer's own end-of-utterance callback is a backstop, delayed
    /// so any audio still in flight from the render callback lands first.
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            guard let self, self.rendering === utterance else { return }
            self.renderingDidEnd(utterance, turn: self.turn)
        }
    }
}
