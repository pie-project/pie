import Foundation

/// The composer's microphone: speech transcribed into the draft.
///
/// A dictation session ends on trailing silence, but a pause to think is
/// not the user being done, so each finished utterance is kept and the
/// next one started until the user taps done (`finish(into:)`) or cancel.
/// A session that ends having heard nothing is left closed.
///
/// Opening and closing are animated here, where they happen, so the
/// composer crossfades between its text field and the recording strip
/// whoever opens or closes dictation (the mic, done, cancel, read-aloud,
/// a refused permission).
@MainActor
final class DictationController: ObservableObject {

    /// Posted on the main queue, with the controller as the object, when
    /// dictation opens and when it closes. Dictation listens on the raw
    /// microphone, which has no echo cancellation, so the chat does not
    /// read aloud while it is open: the read-aloud voice would be
    /// transcribed into the draft.
    static let activityDidChange = Notification.Name("DictationController.activityDidChange")

    @Published private(set) var isDictating = false {
        didSet {
            guard isDictating != oldValue else { return }
            NotificationCenter.default.post(name: Self.activityDidChange, object: self)
        }
    }
    /// The last microphone level, 0...1. Not published: it changes about
    /// 30 times a second, and each publish re-rendered the whole composer.
    /// The waveform reads `meter` once per display frame instead.
    private(set) var level: Float = 0
    /// What has been heard so far. Not published: nothing on screen shows
    /// it, and it changes with every partial result.
    private(set) var transcript = ""
    /// Seconds since dictation started, published once a second (the clock
    /// shows whole seconds).
    @Published private(set) var elapsed: TimeInterval = 0
    @Published private(set) var availability: VoiceInputAvailability?
    /// The microphone stopped on its own (silence, an error); what was
    /// heard waits for done or cancel. The waveform holds still and dims.
    @Published private(set) var isPaused = false
    /// Done was tapped and the last words are being transcribed (up to
    /// `finishTimeout`); the done button shows a spinner meanwhile.
    @Published private(set) var isFinishing = false

    /// The recent loudness, for the waveform.
    let meter = DictationLevelMeter()

    private let microphone: SpeechInput

    /// Utterances already finalised in this dictation.
    private var committed = ""
    /// The utterance in progress, as last transcribed.
    private var partial = ""
    /// Set by `finish(into:)` while the last finalise is on its way.
    private var target: ChatController?
    /// Bumped by every start and stop, so late callbacks and timers from
    /// an earlier dictation do nothing.
    private var session: UInt64 = 0
    private var ticker: Task<Void, Never>?
    private var finishDeadline: Task<Void, Never>?

    /// How long `finish(into:)` waits for the recogniser's final
    /// transcript before settling for the last partial one.
    private static let finishTimeout: UInt64 = 2_000_000_000

    init(microphone: SpeechInput) {
        self.microphone = microphone
    }

    func start() {
        guard !isDictating else { return }
        session += 1
        let current = session
        committed = ""
        partial = ""
        transcript = ""
        level = 0
        meter.reset()
        target = nil
        setOpen(true)
        microphone.delegate = self
        startTicker()

        Task {
            let availability = await microphone.prepare()
            guard isDictating, session == current else { return }
            // The composer hides the strip while dictation is known to be
            // refused; this eases it in if permission has since been given.
            withMotion(Motion.crossfade) { self.availability = availability }
            guard availability.isReady else {
                reset()
                return
            }
            listen()
        }
    }

    /// Stops and appends what was heard to the chat's draft.
    func finish(into chat: ChatController) {
        guard isDictating, target == nil else { return }
        target = chat
        isFinishing = true
        // The clock stops at the moment done was tapped.
        ticker?.cancel()
        guard microphone.isListening else {
            deliver()
            return
        }
        // The final transcript arrives through the delegate. If it never
        // does, the last partial is still what the user said.
        microphone.stop()
        let current = session
        finishDeadline = Task { [weak self] in
            try? await Task.sleep(nanoseconds: Self.finishTimeout)
            guard let self, !Task.isCancelled, self.session == current, self.isDictating else { return }
            self.microphone.cancel()
            self.commit(self.partial)
            self.deliver()
        }
    }

    func cancel() {
        guard isDictating else { return }
        // Voice mode takes the shared microphone over without telling the
        // session it ended; cancelling then would end voice mode's session.
        if microphone.delegate === self {
            microphone.cancel()
        }
        reset()
    }

    // MARK: - Session

    private func listen() {
        // Only when it changes: a needless publish right after opening (the
        // microphone is often ready within the same frame) would cancel the
        // opening crossfade; see `setOpen(_:)`.
        if isPaused { isPaused = false }
        microphone.delegate = self
        do {
            let opening = Date()
            try microphone.start(mode: .dictation)
            // Opening the microphone blocks the main thread (about 4 s in
            // the Simulator); the phone's real cost lands in the console log.
            print(String(format: "[dictation] microphone opened in %.0f ms", Date().timeIntervalSince(opening) * 1000))
        } catch {
            availability = .unavailable(error.localizedDescription)
            pause()
        }
    }

    /// The microphone stopped on its own (no speech, an error, a permission
    /// revoked); what was heard is kept for done or cancel.
    private func pause() {
        commit(partial)
        level = 0
        meter.report(0)
        isPaused = true
        ticker?.cancel()
        if target != nil { deliver() }
    }

    private func commit(_ text: String) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        partial = ""
        if !text.isEmpty {
            committed = committed.isEmpty ? text : committed + " " + text
        }
        transcript = committed
    }

    private func deliver() {
        if let chat = target, !committed.isEmpty {
            let draft = chat.draft
            if draft.isEmpty || draft.last?.isWhitespace == true {
                chat.draft = draft + committed
            } else {
                chat.draft = draft + " " + committed
            }
        }
        reset()
    }

    private func reset() {
        session += 1
        committed = ""
        partial = ""
        transcript = ""
        level = 0
        target = nil
        ticker?.cancel()
        ticker = nil
        finishDeadline?.cancel()
        finishDeadline = nil
        setOpen(false)
    }

    /// Opens or closes dictation, with every published change of it in one
    /// animated transaction, so the composer crossfades between the field
    /// and the strip. All in one on purpose: measured in the Simulator, a
    /// plain publish from this controller in the same turn (resetting the
    /// clock, say) took the animation away and the strip appeared in a
    /// single frame. `elapsed` is reset on opening only, so a closing clock
    /// does not flash back to 0:00 as it fades.
    private func setOpen(_ open: Bool) {
        withMotion(Motion.crossfade) {
            if open { elapsed = 0 }
            isPaused = false
            isFinishing = false
            isDictating = open
        }
    }

    private func startTicker() {
        ticker?.cancel()
        let started = Date()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 250_000_000)
                guard !Task.isCancelled, let self else { return }
                let elapsed = -started.timeIntervalSinceNow
                // Only when the shown second changes: each publish
                // re-renders the composer.
                if Int(elapsed) != Int(self.elapsed) { self.elapsed = elapsed }
            }
        }
    }
}

// MARK: - SpeechInputDelegate

extension DictationController: @preconcurrency SpeechInputDelegate {
    func speechInputDidChangeAvailability(_ availability: VoiceInputAvailability) {
        self.availability = availability
        if isDictating, !availability.isReady { pause() }
    }

    func speechInputDidUpdatePartial(_ text: String) {
        guard isDictating else { return }
        partial = text
        let heard = text.trimmingCharacters(in: .whitespacesAndNewlines)
        transcript = committed.isEmpty ? heard : committed + " " + heard
    }

    func speechInputDidFinalize(_ text: String) {
        guard isDictating else { return }
        let final = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let heard = final.isEmpty ? partial : final
        commit(heard)
        if target != nil {
            deliver()
        } else if heard.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // The microphone gave up waiting for speech. Reopening it would
            // only wait again; what was heard waits for done or cancel.
            pause()
        } else {
            // Trailing silence ended an utterance, but the user has not
            // said they are done: a pause to think, so keep listening.
            listen()
        }
    }

    func speechInputDidUpdateLevel(_ level: Float) {
        guard isDictating else { return }
        self.level = level
        meter.report(level)
    }

    func speechInputDidDetectSpeechOnset() {}

    func speechInputDidFail(_ error: Error) {
        guard isDictating else { return }
        availability = .unavailable(error.localizedDescription)
        pause()
    }
}
