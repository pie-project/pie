import AVFoundation
import Foundation
import Speech

/// Whether this device can turn speech into text, and on what terms.
///
/// `onDevice` is not a detail — the point of running the model locally is
/// undone if the audio is shipped to a transcription server, so the UI
/// shows which one is in force.
enum VoiceInputAvailability: Equatable {
    case ready(onDevice: Bool)
    case denied(String)
    case unavailable(String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

/// Whether on-device transcription can actually be used here.
///
/// `SFSpeechRecognizer.supportsOnDeviceRecognition` reports the
/// capability, not whether the language assets are installed. Where they
/// are missing — the Simulator, and any device that hasn't downloaded
/// them — an on-device request opens the audio, finalises within
/// milliseconds, and fails with "No speech detected". That is
/// indistinguishable from silence unless you notice it happening every
/// single time, so the first such failure disables the local path
/// process-wide and the UI is told the transcription is no longer local.
enum OnDeviceRecognition {
    private(set) static var isUsable = true

    static func markUnusable() {
        isUsable = false
    }

    static func isAvailable(on recognizer: SFSpeechRecognizer) -> Bool {
        isUsable && recognizer.supportsOnDeviceRecognition
    }
}

protocol VoiceInputDelegate: AnyObject {
    /// The input's availability changed after `prepare()` — currently
    /// only when on-device recognition turns out to be unusable.
    func voiceInputDidChangeAvailability(_ availability: VoiceInputAvailability)

    /// Best transcription so far, revised as the user keeps talking.
    func voiceInputDidUpdatePartial(_ text: String)
    /// The utterance is over; this is what was said.
    func voiceInputDidFinalize(_ text: String)
    /// Normalised 0…1 input level, for the meter.
    func voiceInputDidUpdateLevel(_ level: Float)
    func voiceInputDidFail(_ error: Error)
}

/// A source of user utterances.
///
/// Two implementations ship: the microphone, and a bundled audio file
/// (which is how the pipeline gets exercised in the Simulator, where
/// nobody can speak into it).
protocol VoiceInput: AnyObject {
    var delegate: VoiceInputDelegate? { get set }
    var isListening: Bool { get }

    /// Requests whatever permissions this source needs.
    func prepare() async -> VoiceInputAvailability

    func start() throws
    /// Ends the current utterance and finalises the transcription.
    func stop()

    /// Rewinds any fixed sequence of utterances this source plays, so a
    /// new conversation starts from the same place every time.
    func resetSequence()
}

extension VoiceInput {
    func resetSequence() {}
}

protocol VoiceOutputDelegate: AnyObject {
    func voiceOutputDidStartSpeaking()
    func voiceOutputDidFinishSpeaking()
}

/// A sink that says things out loud.
protocol VoiceOutput: AnyObject {
    var delegate: VoiceOutputDelegate? { get set }
    var isSpeaking: Bool { get }

    /// Enqueues a chunk. Chunks are spoken in order, and speaking starts
    /// as soon as the first one arrives — the caller is expected to feed
    /// sentences while the model is still generating.
    func enqueue(_ text: String)

    /// No more chunks are coming for this turn.
    func finishTurn()

    /// Stops mid-word and drops the queue (barge-in).
    func cancel()
}

/// One place that owns the `AVAudioSession` category, because recording
/// and speaking fight over it otherwise.
enum AudioSessionCoordinator {
    private(set) static var isConfigured = false

    /// Play-and-record for the whole app lifetime: switching categories
    /// between listening and speaking costs a session activation each
    /// turn, which is audible as a gap.
    static func configure() throws {
        guard !isConfigured else { return }
        let session = AVAudioSession.sharedInstance()
        // `.defaultToSpeaker` keeps the reply off the earpiece; the
        // Bluetooth option lets AirPods carry it when they are connected,
        // which play-and-record otherwise refuses to route to.
        try session.setCategory(
            .playAndRecord,
            mode: .spokenAudio,
            options: [.defaultToSpeaker, .duckOthers, .allowBluetoothA2DP]
        )
        try session.setActive(true, options: [])
        isConfigured = true
    }

    static func deactivate() {
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: [.notifyOthersOnDeactivation]
        )
        isConfigured = false
    }
}
