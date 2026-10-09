import Foundation

/// Whether this device can turn speech into text, and on what terms.
///
/// `onDevice` is not a detail: the point of running the model locally is
/// undone if the audio is shipped to a transcription server, so the UI
/// says which one is in force.
enum VoiceInputAvailability: Equatable {
    case ready(onDevice: Bool)
    case denied(String)
    case unavailable(String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

/// What a listening session is for.
enum ListeningMode: Equatable {
    /// One utterance for the composer's dictation button. Ends on trailing
    /// silence or `stop()`.
    case dictation
    /// Voice mode. Echo-cancelled, so the reply playing through the
    /// speaker is not heard as the user. Nothing is transcribed until the
    /// user starts talking (`speechInputDidDetectSpeechOnset`); an utterance
    /// ends on trailing silence. May run while speech output is playing,
    /// which is how the user talks over a reply.
    case conversation
}

protocol SpeechInputDelegate: AnyObject {
    /// Availability changed after `prepare()`, for example when on-device
    /// recognition turns out to be unusable.
    func speechInputDidChangeAvailability(_ availability: VoiceInputAvailability)
    /// Best transcription so far, revised as the user keeps talking.
    func speechInputDidUpdatePartial(_ text: String)
    /// The utterance is over; this is what was said (possibly empty).
    func speechInputDidFinalize(_ text: String)
    /// Normalised 0...1 input level, about 20-30 times a second.
    func speechInputDidUpdateLevel(_ level: Float)
    /// The user started talking. Fires once per utterance. In voice mode
    /// this is the barge-in signal while a reply is playing.
    func speechInputDidDetectSpeechOnset()
    func speechInputDidFail(_ error: Error)
}

/// A source of user utterances. All delegate calls arrive on the main
/// queue. One delegate at a time: whoever starts a session sets itself as
/// the delegate first.
protocol SpeechInput: AnyObject {
    var delegate: SpeechInputDelegate? { get set }
    var isListening: Bool { get }

    /// Requests whatever permissions this source needs.
    func prepare() async -> VoiceInputAvailability

    func start(mode: ListeningMode) throws
    /// Ends the current utterance and finalises its transcription.
    func stop()
    /// Ends the session and discards it: no finalise callback.
    func cancel()
}

protocol SpeechOutputDelegate: AnyObject {
    func speechOutputDidStart()
    /// The queue drained (`interrupted == false`) or `stop()` cut it off.
    func speechOutputDidFinish(interrupted: Bool)
    /// Normalised 0...1 level of what is being played, about 30 times a
    /// second while speaking, for the voice-mode orb.
    func speechOutputLevel(_ level: Float)
}

/// A sink that says things out loud. All delegate calls arrive on the
/// main queue. One delegate at a time, as with `SpeechInput`.
protocol SpeechOutput: AnyObject {
    var delegate: SpeechOutputDelegate? { get set }
    /// True from the first `enqueue` of a turn until the queue drains or
    /// `stop()` is called.
    var isSpeaking: Bool { get }
    /// An `AVSpeechSynthesisVoice` identifier, or nil for the best
    /// installed voice in the current language.
    var voiceIdentifier: String? { get set }
    /// In `AVSpeechUtterance` rate units (0...1, default 0.5).
    var rate: Float { get set }

    /// Queues a chunk of text. Chunks are spoken in order, and speaking
    /// starts as soon as the first arrives: callers feed sentences while
    /// the model is still generating.
    func enqueue(_ text: String)
    /// No more chunks are coming for this turn.
    func finishTurn()
    /// Silences playback at once (within about 100 ms) and drops the queue.
    func stop()
    /// Plays at a low volume while true. Voice mode ducks a reply the
    /// moment the microphone hears something, so a user starting to talk
    /// over it is not drowned out, without cutting the reply off on what
    /// may only be a cough or the room. Cleared when a new turn starts.
    var isDucked: Bool { get set }
}

/// An installed text-to-speech voice, for the settings picker.
struct SpeechVoiceInfo: Identifiable, Equatable {
    /// The `AVSpeechSynthesisVoice` identifier.
    let id: String
    let name: String
    /// "Premium", "Enhanced" or "Default".
    let quality: String
    let language: String
}
