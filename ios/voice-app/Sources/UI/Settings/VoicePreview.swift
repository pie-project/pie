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

    /// Plays the sample from the start, cutting off one already playing.
    func play() {
        // A message being read aloud is stopped through its controller, so
        // the chat's speaker button does not stay lit for speech that has
        // been taken over.
        chat.stopReadingAloud()
        if isPlaying {
            speech.stop()
        }
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

    func toggle() {
        if isPlaying { stop() } else { play() }
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
