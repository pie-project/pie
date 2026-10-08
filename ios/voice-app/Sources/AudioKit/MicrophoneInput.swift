import AVFoundation
import Foundation
import Speech

/// `VoiceInput` from the microphone, transcribed by the Speech framework.
///
/// On-device recognition is requested whenever the device supports it, so
/// that in the intended configuration nothing spoken to this app leaves
/// the phone — the audio is transcribed locally and the transcript is
/// answered by a model that is also local. When on-device recognition is
/// unavailable the class still works, but it reports that fact rather
/// than quietly shipping audio to a server.
final class MicrophoneInput: NSObject, VoiceInput {

    weak var delegate: VoiceInputDelegate?
    private(set) var isListening = false

    /// Silence after speech that ends an utterance. Long enough to think
    /// mid-sentence, short enough that the app doesn't feel deaf.
    private let endOfSpeechSilence: TimeInterval = 1.3

    /// An on-device request whose language assets are missing fails
    /// within milliseconds of opening the audio. Anything that dies this
    /// soon, unprompted, is that failure and not the user's silence.
    private let assetFailureWindow: TimeInterval = 3.0

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let audioEngine = AVAudioEngine()

    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var silenceTimer: Timer?

    private var latestTranscript = ""
    private var didFinalize = false
    private var usesOnDeviceRecognition = false
    /// Set once `stop()` has been asked for, so a recogniser error that
    /// follows the end of audio reads as "utterance over", not "broken".
    private var stopRequested = false
    private var listeningSince = Date.distantPast
    private var permissionsGranted = false
    /// Bumped by every start() and stop(). Deferred callbacks capture it
    /// and bail if the session they belong to is over, so the 1 s
    /// finalise fallback from one press can't tear down the next one.
    private var generation = 0

    /// On-device recognition never errors on silence, so without this a
    /// press that hears nothing would listen until the 1-minute system
    /// limit. Long enough for someone to gather a thought.
    private let noSpeechTimeout: TimeInterval = 8.0

    override init() {
        super.init()
        recognizer?.delegate = self
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

    // MARK: - VoiceInput

    func prepare() async -> VoiceInputAvailability {
        guard let recognizer else {
            return .unavailable("No speech recogniser for this locale")
        }

        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
        }
        guard speechStatus == .authorized else {
            return .denied("Speech recognition permission was declined — type instead, or allow it in Settings")
        }

        let micGranted = await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { continuation.resume(returning: $0) }
        }
        guard micGranted else {
            return .denied("Microphone permission was declined — type instead, or allow it in Settings")
        }
        permissionsGranted = true

        guard recognizer.isAvailable else {
            return .unavailable("Speech recogniser is not available right now — try again in a moment, or type")
        }

        usesOnDeviceRecognition = OnDeviceRecognition.isAvailable(on: recognizer)
        return .ready(onDevice: usesOnDeviceRecognition)
    }

    func start() throws {
        guard !isListening else { return }
        guard let recognizer, recognizer.isAvailable else {
            throw VoiceInputError.recogniserUnavailable
        }

        try AudioSessionCoordinator.configure()

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = usesOnDeviceRecognition
        request.taskHint = .dictation
        self.request = request

        latestTranscript = ""
        didFinalize = false
        stopRequested = false
        listeningSince = Date()
        generation += 1

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        // A zero-channel or zero-rate format means the session handed us
        // no input route (Simulator without a mic, a Bluetooth handoff in
        // progress). Installing a tap on it crashes; throwing doesn't.
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw VoiceInputError.noAudioInput
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.request?.append(buffer)
            self?.reportLevel(of: buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw error
        }
        isListening = true
        armSilenceTimer(after: noSpeechTimeout)

        task = recognizer.recognitionTask(with: request) { [weak self, weak request] result, error in
            guard let self else { return }
            // A cancelled task delivers one last callback; if a new
            // session has since started (the on-device -> network
            // fallback), that callback must not touch it.
            guard let request, request === self.request else { return }

            if let result {
                let text = result.bestTranscription.formattedString
                if !text.isEmpty {
                    self.latestTranscript = text
                    DispatchQueue.main.async { self.delegate?.voiceInputDidUpdatePartial(text) }
                    self.restartSilenceTimer()
                }
                if result.isFinal {
                    self.finishUtterance()
                    return
                }
            }

            if let error {
                self.handleRecognitionError(error)
            }
        }
    }

    func stop() {
        guard isListening else { return }
        stopRequested = true
        generation += 1
        let stopping = generation
        silenceTimer?.invalidate()
        silenceTimer = nil
        teardownAudio()
        request?.endAudio()

        // The final result usually follows within a few hundred
        // milliseconds. If it doesn't, the last partial is what was said.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            guard let self, self.generation == stopping else { return }
            self.finishUtterance()
        }
    }

    // MARK: - Internals

    private func handleRecognitionError(_ error: Error) {
        // A recogniser that times out after the user has already said
        // something is not a failure — it is the end of the utterance,
        // and the transcript stands.
        if !latestTranscript.isEmpty {
            finishUtterance()
            return
        }
        // Audio was ended on purpose and nothing was said: that is
        // silence, and silence is not an error.
        if stopRequested {
            finishUtterance()
            return
        }
        let diedEarly = -listeningSince.timeIntervalSinceNow < assetFailureWindow
        if usesOnDeviceRecognition && diedEarly {
            // The local recogniser has no usable assets. Live audio can't
            // be replayed, but the microphone can be reopened straight
            // away over the network path — the user just keeps talking.
            OnDeviceRecognition.markUnusable()
            usesOnDeviceRecognition = false
            teardown()
            DispatchQueue.main.async {
                self.delegate?.voiceInputDidChangeAvailability(.ready(onDevice: false))
                do {
                    try self.start()
                } catch {
                    self.delegate?.voiceInputDidFail(error)
                }
            }
            return
        }
        DispatchQueue.main.async { self.delegate?.voiceInputDidFail(error) }
        teardown()
    }

    private func restartSilenceTimer() {
        armSilenceTimer(after: endOfSpeechSilence)
    }

    private func armSilenceTimer(after interval: TimeInterval) {
        DispatchQueue.main.async { [weak self] in
            guard let self, self.isListening else { return }
            self.silenceTimer?.invalidate()
            self.silenceTimer = Timer.scheduledTimer(
                withTimeInterval: interval,
                repeats: false
            ) { [weak self] _ in
                self?.stop()
            }
        }
    }

    /// Siri, a call, or an alarm stops the audio engine underneath us; the
    /// recogniser never finalises on its own. End the utterance with
    /// whatever was heard so the app doesn't sit in "listening…" forever.
    @objc private func audioSessionInterrupted(_ note: Notification) {
        guard
            let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
            AVAudioSession.InterruptionType(rawValue: raw) == .began
        else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self, self.isListening else { return }
            self.stop()
        }
    }

    private func finishUtterance() {
        guard !didFinalize else { return }
        didFinalize = true

        let transcript = latestTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        teardown()
        DispatchQueue.main.async { [weak self] in
            self?.delegate?.voiceInputDidFinalize(transcript)
        }
    }

    private func teardownAudio() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        isListening = false
    }

    private func teardown() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        teardownAudio()
        task?.cancel()
        task = nil
        request = nil
    }

    /// RMS of the buffer, mapped onto 0…1 for the meter ring.
    private func reportLevel(of buffer: AVAudioPCMBuffer) {
        guard let channel = buffer.floatChannelData?[0] else { return }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return }

        var sum: Float = 0
        for index in 0..<count {
            let sample = channel[index]
            sum += sample * sample
        }
        let rms = sqrt(sum / Float(count))
        // -50 dBFS is a quiet room, 0 dBFS is clipping.
        let decibels = 20 * log10(max(rms, 1e-7))
        let level = max(0, min(1, (decibels + 50) / 50))

        DispatchQueue.main.async { [weak self] in
            self?.delegate?.voiceInputDidUpdateLevel(level)
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
            availability = .ready(onDevice: usesOnDeviceRecognition)
        } else {
            availability = .unavailable("Speech recogniser is not available right now — try again in a moment, or type")
        }
        DispatchQueue.main.async { [weak self] in
            self?.delegate?.voiceInputDidChangeAvailability(availability)
        }
    }
}

enum VoiceInputError: LocalizedError {
    case recogniserUnavailable
    case missingAudioFile(String)
    case noAudioInput

    var errorDescription: String? {
        switch self {
        case .recogniserUnavailable:
            return "The speech recogniser is unavailable — try again, or type"
        case .missingAudioFile(let name):
            return "Bundled audio file not found: \(name)"
        case .noAudioInput:
            return "No microphone input is available right now"
        }
    }
}
