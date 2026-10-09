import AVFoundation
import Foundation
import Speech

/// Whether on-device transcription can actually be used here.
///
/// `SFSpeechRecognizer.supportsOnDeviceRecognition` reports the
/// capability, not whether the language assets are installed. Where they
/// are missing (the Simulator, and any device that has not downloaded
/// them) an on-device request opens the audio, finalises within
/// milliseconds, and fails with "No speech detected". That is
/// indistinguishable from silence unless you notice it happening every
/// single time, so the first such failure disables the local path
/// process-wide and the UI is told the transcription is no longer local.
/// Main thread only.
enum OnDeviceRecognition {
    private(set) static var isUsable = true

    /// An on-device request that dies sooner than this after it started,
    /// having heard nothing, failed for want of assets rather than speech.
    static let assetFailureWindow: TimeInterval = 3.0

    static func markUnusable() {
        if isUsable {
            print("[audio] on-device recognition unusable here; falling back to the network recogniser")
        }
        isUsable = false
    }

    static func isAvailable(on recognizer: SFSpeechRecognizer) -> Bool {
        isUsable && recognizer.supportsOnDeviceRecognition
    }
}

enum VoiceInputError: LocalizedError {
    case recogniserUnavailable
    case missingAudioFile(String)
    case noAudioInput
    case interrupted
    case inputLost

    var errorDescription: String? {
        switch self {
        case .recogniserUnavailable:
            return "The speech recogniser is unavailable. Try again, or type."
        case .missingAudioFile(let name):
            return "Bundled audio file not found: \(name)"
        case .noAudioInput:
            return "No microphone input is available right now."
        case .interrupted:
            return "Listening stopped because another app or a call took the audio. Tap to resume."
        case .inputLost:
            return "Listening stopped because the microphone went away when the audio route changed. Tap to resume."
        }
    }
}

enum SpeechPermissions {
    static func requestRecognition() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }

    static func requestMicrophone() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }
}

enum SpeechRecognition {

    /// The recogniser both inputs use. The bundled model and the sample
    /// recordings are English, so the recogniser is too.
    static func makeRecognizer() -> SFSpeechRecognizer? {
        SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    }

    static func configure(_ request: SFSpeechRecognitionRequest, onDevice: Bool) {
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = onDevice
        request.taskHint = .dictation
        // The model reads the transcript; punctuation tells it where the
        // question ends.
        request.addsPunctuation = true
    }

    static func bufferRequest(onDevice: Bool) -> SFSpeechAudioBufferRecognitionRequest {
        let request = SFSpeechAudioBufferRecognitionRequest()
        configure(request, onDevice: onDevice)
        return request
    }

    /// "No speech detected": the recogniser heard nothing it could
    /// transcribe. After an onset that was a cough or a door, that is an
    /// empty utterance, not a broken microphone.
    static func isNoSpeech(_ error: Error) -> Bool {
        let error = error as NSError
        return error.domain == "kAFAssistantErrorDomain" && error.code == 1110
    }

    static let unavailableMessage =
        "Speech recognition is not available right now. Try again in a moment, or type."
}
