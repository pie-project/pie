import AVFoundation
import Foundation

/// `VoiceOutput` on `AVSpeechSynthesizer`.
///
/// The synthesizer already queues utterances and plays them in order, so
/// enqueuing sentences as they generate is enough to overlap speech with
/// generation. This class exists to own the voice choice, translate
/// delegate callbacks into the app's own protocol, and report "speaking"
/// as a single state rather than per-utterance.
final class SpokenOutput: NSObject, VoiceOutput {

    weak var delegate: VoiceOutputDelegate?

    private let synthesizer = AVSpeechSynthesizer()
    private var pendingUtterances = 0
    private var turnIsOpen = false

    /// True from the first enqueue of a turn until the queue drains. Set
    /// on enqueue rather than on the synthesizer's asynchronous `didStart`,
    /// so a caller that checks it right after enqueuing the last sentence
    /// sees speech pending instead of declaring the turn over.
    private(set) var isSpeaking = false

    override init() {
        super.init()
        synthesizer.delegate = self
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(audioSessionInterrupted(_:)),
            name: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance()
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - VoiceOutput

    func enqueue(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        // A typed question never touches the microphone, so nothing else
        // has configured the audio session: the synthesizer would speak
        // through the default ambient session, which the ring/silent
        // switch mutes. Claim the shared play-and-record session here so
        // the reply is audible either way.
        try? AudioSessionCoordinator.configure()

        turnIsOpen = true
        pendingUtterances += 1

        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = Self.preferredVoice
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0
        // A beat between sentences; without it the queue runs them
        // together and the reply sounds breathless.
        utterance.postUtteranceDelay = 0.05

        if !isSpeaking {
            isSpeaking = true
            delegate?.voiceOutputDidStartSpeaking()
        }
        synthesizer.speak(utterance)
    }

    func finishTurn() {
        turnIsOpen = false
        // If the queue already drained, the final callback has been and
        // gone — settle the state here instead of waiting for one that
        // will never arrive.
        if pendingUtterances == 0 {
            settleIfIdle()
        }
    }

    func cancel() {
        turnIsOpen = false
        pendingUtterances = 0
        synthesizer.stopSpeaking(at: .immediate)
        settleIfIdle()
    }

    // MARK: - State

    private func settleIfIdle() {
        guard isSpeaking, pendingUtterances == 0, !turnIsOpen else { return }
        isSpeaking = false
        delegate?.voiceOutputDidFinishSpeaking()
    }

    /// A phone call, Siri, or an alarm takes the audio session away
    /// mid-sentence; the synthesizer stops but its callbacks may never
    /// come. Treat it as a cancel so the app settles instead of sitting in
    /// "speaking" forever.
    @objc private func audioSessionInterrupted(_ note: Notification) {
        guard
            let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
            AVAudioSession.InterruptionType(rawValue: raw) == .began
        else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self, self.isSpeaking else { return }
            self.cancel()
        }
    }

    /// Prefers an enhanced/premium voice when the device has one
    /// downloaded; the compact default is noticeably robotic, and a demo
    /// of a talking phone is judged partly on how it sounds.
    private static let preferredVoice: AVSpeechSynthesisVoice? = {
        let language = AVSpeechSynthesisVoice.currentLanguageCode()
        let candidates = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language == language }

        let ranked = candidates.sorted { lhs, rhs in
            SpokenOutput.rank(lhs) > SpokenOutput.rank(rhs)
        }
        return ranked.first ?? AVSpeechSynthesisVoice(language: language)
    }()

    private static func rank(_ voice: AVSpeechSynthesisVoice) -> Int {
        switch voice.quality {
        case .premium: return 3
        case .enhanced: return 2
        default: return 1
        }
    }
}

extension SpokenOutput: AVSpeechSynthesizerDelegate {

    func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didStart utterance: AVSpeechUtterance
    ) {
        guard !isSpeaking else { return }
        isSpeaking = true
        delegate?.voiceOutputDidStartSpeaking()
    }

    func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        pendingUtterances = max(0, pendingUtterances - 1)
        settleIfIdle()
    }

    func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didCancel utterance: AVSpeechUtterance
    ) {
        pendingUtterances = max(0, pendingUtterances - 1)
        settleIfIdle()
    }
}
