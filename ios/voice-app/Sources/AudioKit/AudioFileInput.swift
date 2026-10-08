import AVFoundation
import Foundation
import Speech

/// `VoiceInput` that transcribes a bundled recording instead of the
/// microphone.
///
/// The Simulator has no one to talk to it, so this is how the voice path
/// gets exercised end to end without a human in the loop: the same
/// recogniser, the same delegate callbacks, the same controller, the same
/// model — only the source of the audio differs. It is also a repeatable
/// way to compare turns, since the utterance is byte-identical every run.
///
/// That this drops in behind `VoiceInput` with no change above it is the
/// modularity claim, tested rather than asserted.
final class AudioFileInput: NSObject, VoiceInput {

    weak var delegate: VoiceInputDelegate?
    private(set) var isListening = false

    /// Played in order, one per press, then wrapped around. The later
    /// recordings are deliberately follow-ups ("say that more simply")
    /// that only make sense if the conversation carried over — which is
    /// what makes this a test of the backend's session state and not just
    /// of the audio path.
    private let resources: [String]
    private var index = 0
    private let fileExtension: String
    /// Plays the recording aloud while transcribing, so a screen capture
    /// of a demo has the question audible on it.
    private let playsAudibly: Bool

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var task: SFSpeechRecognitionTask?
    private var player: AVAudioPlayer?
    private var latestTranscript = ""
    private var didFinalize = false
    private var usesOnDeviceRecognition = false

    init(resources: [String], fileExtension: String = "wav", playsAudibly: Bool = true) {
        self.resources = resources
        self.fileExtension = fileExtension
        self.playsAudibly = playsAudibly
    }

    /// The recording the next `start()` will use.
    var url: URL? {
        guard !resources.isEmpty else { return nil }
        return Bundle.main.url(
            forResource: resources[index % resources.count],
            withExtension: fileExtension
        )
    }

    // MARK: - VoiceInput

    func prepare() async -> VoiceInputAvailability {
        guard let recognizer else {
            return .unavailable("No speech recogniser for this locale")
        }
        guard url != nil else {
            return .unavailable("No bundled sample recordings were found")
        }

        let status = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
        }
        guard status == .authorized else {
            return .denied("Speech recognition permission was declined")
        }
        guard recognizer.isAvailable else {
            return .unavailable("Speech recogniser is not available right now")
        }

        usesOnDeviceRecognition = OnDeviceRecognition.isAvailable(on: recognizer)
        return .ready(onDevice: usesOnDeviceRecognition)
    }

    func start() throws {
        guard !isListening else { return }
        guard let recognizer, recognizer.isAvailable else {
            throw VoiceInputError.recogniserUnavailable
        }
        guard let url else {
            throw VoiceInputError.missingAudioFile(
                "\(resources.first ?? "sample").\(fileExtension)"
            )
        }

        try AudioSessionCoordinator.configure()

        latestTranscript = ""
        didFinalize = false
        isListening = true

        if playsAudibly {
            player = try? AVAudioPlayer(contentsOf: url)
            player?.play()
        }

        let request = SFSpeechURLRecognitionRequest(url: url)
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = usesOnDeviceRecognition
        request.taskHint = .dictation

        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }

            if let result {
                let text = result.bestTranscription.formattedString
                if !text.isEmpty {
                    self.latestTranscript = text
                    DispatchQueue.main.async { self.delegate?.voiceInputDidUpdatePartial(text) }
                }
                if result.isFinal {
                    self.finishUtterance()
                    return
                }
            }

            if let error {
                if !self.latestTranscript.isEmpty {
                    self.finishUtterance()
                    return
                }
                // A local recogniser that produces nothing at all is
                // almost certainly missing its language assets rather
                // than looking at silence. The audio is a file, so the
                // attempt can simply be made again over the network.
                if self.usesOnDeviceRecognition {
                    OnDeviceRecognition.markUnusable()
                    self.usesOnDeviceRecognition = false
                    self.isListening = false
                    DispatchQueue.main.async {
                        self.delegate?.voiceInputDidChangeAvailability(.ready(onDevice: false))
                        try? self.start()
                    }
                    return
                }
                self.isListening = false
                DispatchQueue.main.async { self.delegate?.voiceInputDidFail(error) }
            }
        }
    }

    func stop() {
        guard isListening else { return }
        finishUtterance()
    }

    func resetSequence() {
        index = 0
    }

    private func finishUtterance() {
        guard !didFinalize else { return }
        didFinalize = true
        isListening = false

        task?.cancel()
        task = nil
        index += 1

        let transcript = latestTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        DispatchQueue.main.async { [weak self] in
            self?.delegate?.voiceInputDidFinalize(transcript)
        }
    }
}
