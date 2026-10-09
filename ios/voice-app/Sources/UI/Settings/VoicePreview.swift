import Foundation

/// Speaks a short sample through the app's own synthesizer, so the voice
/// and rate chosen in Settings are heard exactly as voice mode and
/// read-aloud will use them.
@MainActor
final class VoicePreview: ObservableObject, @preconcurrency SpeechOutputDelegate {

    @Published private(set) var isPlaying = false

    private let speech: SpeechOutput
    private let chat: ChatController

    static let sampleSentence =
        "Hi, I'm Pie. I run entirely on this iPhone, so what you say to me stays here."

    init(speech: SpeechOutput, chat: ChatController) {
        self.speech = speech
        self.chat = chat
    }

    /// Plays the sample from the start in the given voice (nil for
    /// Automatic) and rate, cutting off whatever is playing.
    ///
    /// The voice and rate are applied here rather than left to the
    /// settings subscription, so the sample is always the one on screen.
    /// The synthesizer fixes a turn's voice when the turn starts, so any
    /// speech still going is stopped first: text added to it would come
    /// out in the old voice.
    func play(voiceIdentifier: String?, rate: Float) {
        // A message being read aloud is stopped through its controller, so
        // the chat's speaker button does not stay lit for speech that has
        // been taken over.
        chat.stopReadingAloud()
        if isPlaying || speech.isSpeaking {
            speech.stop()
        }
        speech.voiceIdentifier = voiceIdentifier
        speech.rate = rate
        speech.delegate = self
        speech.enqueue(Self.sampleSentence)
        speech.finishTurn()
        isPlaying = true
    }

    func stop() {
        guard isPlaying else { return }
        isPlaying = false
        speech.stop()
    }

    func toggle(voiceIdentifier: String?, rate: Float) {
        if isPlaying {
            stop()
        } else {
            play(voiceIdentifier: voiceIdentifier, rate: rate)
        }
    }

    // MARK: - SpeechOutputDelegate

    func speechOutputDidStart() {}

    func speechOutputDidFinish(interrupted: Bool) {
        // The report for speech cut off to start this sample can arrive
        // after the sample has begun; it is not this sample's end.
        guard !speech.isSpeaking else { return }
        isPlaying = false
    }

    func speechOutputLevel(_ level: Float) {}
}
