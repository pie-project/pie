import AVFoundation
import Foundation

/// On-device checks of the audio path, run with `-PieAudioCheck 1`: the
/// sample question finishes playing before the reply speaks, `stop()`
/// silences speech promptly, a cancelled reply frees the engine for the
/// next one, and echo cancellation keeps a playing reply from triggering
/// a false barge-in. Prints `PIECHECK {json}` lines and writes
/// Documents/audio-check.jsonl, then exits.
///
/// None of this can be exercised on a Mac: voice processing, the speaker
/// route and the recogniser's assets only exist on the phone, so the
/// checks measure them there instead of assuming. Timing is read from the
/// hub's playback channels themselves (what is queued, and the level of
/// what is leaving the speaker) and from the microphone, never from the
/// bookkeeping of the code under test, so a check can fail. Every line
/// carries `check` and `pass` (true, false, or null for informational and
/// skipped checks); times are seconds. Nothing waits without a limit, so
/// the run ends on its own even with a permission prompt left unanswered.
/// The checks drive only the audio and backend protocols plus the
/// AudioKit types, never the UI.
enum AudioSelfCheck {

    static var isEnabled: Bool {
        UserDefaults.standard.string(forKey: "PieAudioCheck") == "1"
    }

    @MainActor
    static func run(
        backend: ConversationBackend,
        speech: SpeechOutput,
        microphone: SpeechInput,
        sample: SampleQuestionInput
    ) {
        let checks = AudioChecks(backend: backend, speech: speech, microphone: microphone, sample: sample)
        Task { @MainActor in
            await checks.runAll()
            exit(0)
        }
    }
}

@MainActor
private final class AudioChecks {

    private static let session = "audio-check"
    private static let voicePrompt =
        "You are Pie, talking out loud from this iPhone. "
        + "Answer in one or two short spoken sentences, with no lists or formatting."
    private static let voiceOptions = ReplyOptions(maxTokens: 60, temperature: 0.7, topP: 0.95, think: false)
    /// Comfortably longer than any wait below, so speech is still going
    /// when it is stopped or talked over.
    private static let longParagraph = [
        "The lighthouse stood on a rocky point at the far edge of the bay.",
        "Every evening the keeper climbed the spiral stairs to light the great lamp.",
        "Ships far out at sea watched for its steady beam and knew where the rocks were.",
        "On stormy nights he stayed awake until dawn, listening to the wind against the glass.",
        "In the morning he wrote the weather in his logbook and made a pot of strong tea.",
    ]
    /// About six seconds at the default rate.
    private static let shortParagraph = [
        "The lighthouse stood on a rocky point at the edge of the bay,",
        "and every night its lamp swept slowly across the water.",
    ]
    private static let audible: Float = 0.02

    private let backend: ConversationBackend
    private let speech: SpeechOutput
    private let microphone: SpeechInput
    private let sample: SampleQuestionInput
    private let probe = Probe()
    private let log = CheckLog()
    private var verdicts: [(name: String, pass: Bool?)] = []

    init(backend: ConversationBackend, speech: SpeechOutput, microphone: SpeechInput, sample: SampleQuestionInput) {
        self.backend = backend
        self.speech = speech
        self.microphone = microphone
        self.sample = sample
    }

    func runAll() async {
        let began = Date()
        log.emit(["check": "begin", "pass": NSNull(), "date": ISO8601DateFormatter().string(from: began)])

        let engineReady = await warmUp()
        let microphoneAvailability = await prepare(microphone)
        await sampleOverlap(engineReady: engineReady, microphoneReady: microphoneAvailability.isReady)
        await speechStopLatency(microphoneReady: microphoneAvailability.isReady)
        await cancelFreesEngine(engineReady: engineReady)
        await noFalseBargeIn(microphoneAvailability)
        await externalOnset(microphoneAvailability)

        var results: [String: Any] = [:]
        for verdict in verdicts {
            results[verdict.name] = verdict.pass.map { $0 as Any } ?? NSNull()
        }
        let failed = verdicts.contains { $0.pass == false }
        log.emit([
            "check": "summary",
            "pass": !failed,
            "results": results,
            "seconds": Self.rounded(Date().timeIntervalSince(began)),
        ])
    }

    // MARK: - a. warmup

    private func warmUp() async -> Bool {
        let start = Date()
        let outcome = WarmUpOutcome()
        let backend = self.backend
        Task { @MainActor in
            outcome.message = await backend.warmUp()
            outcome.done = true
        }
        let finished = await wait(upTo: 600) { outcome.done }
        let ok = finished && outcome.message == nil
        record("warmup", pass: ok, [
            "seconds": Self.rounded(Date().timeIntervalSince(start)),
            "error": outcome.message ?? (finished ? NSNull() : "timed out after 600 s" as Any),
        ])
        return ok
    }

    // MARK: - b. sample_overlap

    /// The user's bug: the reply started talking while the sample question
    /// was still playing. Read from the hub's channels: the recording must
    /// have stopped coming out of the speaker before its transcript is
    /// final and before any of the reply does, and the two channels must
    /// never have audio queued at the same time. In between it does what
    /// voice mode does: the echo-cancelled microphone comes back on, which
    /// rebuilds the engine, as soon as the transcript is final.
    private func sampleOverlap(engineReady: Bool, microphoneReady: Bool) async {
        let name = "sample_overlap"
        let availability = await prepare(sample)
        guard availability.isReady else {
            record(name, pass: nil, ["skipped": "sample input unavailable: \(availability)"])
            return
        }
        probe.reset()
        sample.delegate = probe
        speech.delegate = probe
        sample.resetSequence()
        let channels = ChannelRecorder()
        channels.start()
        defer { channels.stop() }

        let start = Date()
        do {
            try sample.start(mode: .conversation)
        } catch {
            record(name, pass: false, ["error": error.localizedDescription])
            return
        }
        let finalized = await wait(upTo: 30) { !self.probe.finals.isEmpty || !self.probe.failures.isEmpty }
        guard finalized, let final = probe.finals.first else {
            sample.cancel()
            record(name, pass: false, [
                "error": probe.failures.first ?? "no transcript within 30 s",
                "on_device": "\(probe.availability)",
            ])
            return
        }

        // What voice mode does next.
        let listener = Probe()
        var microphoneOn = false
        var fields: [String: Any] = [:]
        if microphoneReady {
            microphone.delegate = listener
            do {
                try microphone.start(mode: .conversation)
                microphoneOn = true
            } catch {
                fields["microphone_error"] = error.localizedDescription
            }
        }

        let cueLastAudible = channels.samples.last { $0.cueLevel >= Self.audible }?.at
        let cueAfterFinal = channels.samples.contains { $0.at > final.at && $0.cueSounding }
        let finalizedAfterPlayback = cueLastAudible.map { final.at >= $0 } ?? false
        fields["transcript"] = final.text
        fields["onset"] = Self.seconds(start, probe.onsets.first)
        fields["cue_last_audible"] = Self.seconds(start, cueLastAudible)
        fields["finalize"] = Self.seconds(start, final.at)
        fields["cue_after_finalize"] = cueAfterFinal
        fields["availability_changes"] = probe.availability
        fields["microphone_on"] = microphoneOn

        guard engineReady else {
            if microphoneOn { microphone.cancel() }
            fields["note"] = "engine not ready, so the reply was not generated"
            record(name, pass: finalizedAfterPlayback && !cueAfterFinal, fields)
            return
        }

        let chunker = SentenceChunker()
        let speech = self.speech
        let reply = await runReply(
            ReplyTicket(),
            messages: [
                PromptMessage(role: .system, content: Self.voicePrompt),
                PromptMessage(role: .user, content: final.text),
            ],
            options: Self.voiceOptions,
            timeout: 90
        ) { event in
            guard case .text(let delta) = event else { return }
            for sentence in chunker.push(delta) {
                speech.enqueue(sentence)
            }
        }
        if let rest = chunker.flush() {
            speech.enqueue(rest)
        }
        speech.finishTurn()
        if !probe.speechStarts.isEmpty || speech.isSpeaking {
            _ = await wait(upTo: 60) { !self.probe.speechFinishes.isEmpty }
        }
        if microphoneOn { microphone.cancel() }

        let firstAudible = channels.samples.first { $0.speechLevel >= Self.audible }
        let cueLastSounding = channels.samples.last { $0.cueSounding }?.at
        let overlapping = channels.samples.filter { $0.cueSounding && $0.speechSounding }.count
        fields["reply"] = reply.result?.text ?? reply.error ?? "no reply within 90 s"
        fields["speech_start"] = Self.seconds(start, probe.speechStarts.first)
        fields["first_audible"] = Self.seconds(start, firstAudible?.at)
        fields["cue_at_first_audible"] = firstAudible.map {
            ["queued": $0.cueActive, "level": Self.rounded(TimeInterval($0.cueLevel))] as [String: Any]
        } ?? NSNull()
        fields["overlapping_samples"] = overlapping
        if microphoneOn {
            // Informational here (no_false_bargein is the verdict): onsets
            // the reply's own residue caused right after the rebuild, which
            // voice mode would have taken as the user interrupting.
            fields["microphone_onsets_during_reply"] = listener.onsets.filter { $0 >= final.at }.count
            fields["residue_during_reply"] = listener.inputDecibels(from: firstAudible?.at ?? final.at, to: Date())
        }
        let spokeAfterPlayback: Bool
        if let firstAudible {
            spokeAfterPlayback = !firstAudible.cueSounding && (cueLastSounding.map { $0 < firstAudible.at } ?? true)
        } else {
            spokeAfterPlayback = false
        }
        record(name, pass: finalizedAfterPlayback && !cueAfterFinal && spokeAfterPlayback && overlapping == 0, fields)
    }

    // MARK: - c. speech_stop_latency

    /// `stop()` must silence speech within 150 ms, and nothing from the
    /// stopped turn may play afterwards. Read three ways: the synthesizer's
    /// finish report; the speech channel itself (queued audio and the
    /// level leaving the speaker, every 10 ms); and the plain microphone
    /// (dictation mode, no echo cancellation), which hears the speaker as
    /// it is and is the one witness independent of the app's bookkeeping.
    /// The microphone reports about one tap buffer (~100 ms) late, so its
    /// limit is looser.
    private func speechStopLatency(microphoneReady: Bool) async {
        let name = "speech_stop_latency"
        probe.reset()
        speech.delegate = probe
        let witness = Probe()
        var witnessOn = false
        var fields: [String: Any] = [:]
        if microphoneReady {
            microphone.delegate = witness
            do {
                try microphone.start(mode: .dictation)
                witnessOn = true
                // The room, before anything plays.
                await pause(0.6)
            } catch {
                fields["microphone_error"] = error.localizedDescription
            }
        }
        defer {
            if witnessOn { microphone.cancel() }
        }
        let room = witness.medianDecibels(from: Date().addingTimeInterval(-0.5), to: Date())

        let channels = ChannelRecorder()
        channels.start()
        defer { channels.stop() }
        let start = Date()
        for sentence in Self.longParagraph {
            speech.enqueue(sentence)
        }
        speech.finishTurn()

        let firstAudible = { channels.samples.first { $0.at >= start && $0.speechLevel >= Self.audible }?.at }
        guard await wait(upTo: 10, until: { firstAudible() != nil }), let audibleAt = firstAudible() else {
            speech.stop()
            fields["error"] = "speech never became audible"
            fields["started"] = !probe.speechStarts.isEmpty
            record(name, pass: false, fields)
            return
        }
        await pause(max(0, 1.5 - Date().timeIntervalSince(audibleAt)))

        let stopAt = Date()
        speech.stop()
        let stillSpeaking = speech.isSpeaking
        _ = await wait(upTo: 2) { !self.probe.speechFinishes.isEmpty }
        // Anything from the stopped turn that was going to play late
        // would have shown up by now.
        await pause(0.6)
        let endAt = Date()

        let finish = probe.speechFinishes.first
        let toFinish = finish.map { $0.at.timeIntervalSince(stopAt) }
        let afterStop = channels.samples.filter { $0.at >= stopAt }
        let channelQuietAt = afterStop.first { !$0.speechSounding }?.at
        let channelLate = afterStop.contains { $0.at > (channelQuietAt ?? stopAt) && $0.speechSounding }
        let toChannelQuiet = channelQuietAt.map { $0.timeIntervalSince(stopAt) }
        var pass = finish?.interrupted == true
            && (toFinish ?? .infinity) < 0.15
            && (toChannelQuiet ?? .infinity) < 0.15
            && !channelLate
            && !stillSpeaking

        fields["audible_after"] = Self.rounded(audibleAt.timeIntervalSince(start))
        fields["stop_to_finish"] = Self.json(toFinish)
        fields["stop_to_channel_quiet"] = Self.json(toChannelQuiet)
        fields["channel_audio_after_stop"] = channelLate
        fields["interrupted"] = finish.map { $0.interrupted as Any } ?? NSNull()
        fields["is_speaking_after_stop"] = stillSpeaking

        if witnessOn, let room {
            let speaking = witness.medianDecibels(from: audibleAt, to: stopAt) ?? room
            let reports = witness.decibels(from: stopAt, to: endAt)
            // Quiet: back within 6 dB of the room. Late audio: speech-like
            // loudness again after that, past the reports' own spacing.
            let quietAt = reports.first { $0.decibels < room + 6 }?.at
            let late = reports.contains { $0.at > (quietAt ?? stopAt).addingTimeInterval(0.1) && $0.decibels >= room + 10 }
            let toQuiet = quietAt.map { $0.timeIntervalSince(stopAt) }
            // A route the microphone does not hear (headphones) proves
            // nothing either way.
            let heard = speaking >= room + 10
            fields["microphone_room_dbfs"] = Self.tenths(room)
            fields["microphone_speaking_dbfs"] = Self.tenths(speaking)
            fields["microphone_heard_speech"] = heard
            fields["stop_to_microphone_quiet"] = Self.json(toQuiet)
            fields["microphone_audio_after_stop"] = late
            if heard {
                pass = pass && (toQuiet ?? .infinity) < 0.35 && !late
            }
        } else {
            fields["note"] = "measured on the speech channel only; the microphone was unavailable"
        }
        record(name, pass: pass, fields)
    }

    // MARK: - d. cancel_frees_engine

    private func cancelFreesEngine(engineReady: Bool) async {
        let name = "cancel_frees_engine"
        guard engineReady else {
            record(name, pass: nil, ["skipped": "engine not ready"])
            return
        }
        let backend = self.backend

        // A long reply cancelled as soon as it starts talking.
        let longTicket = ReplyTicket()
        let cancelled = Stamp()
        let long = await runReply(
            longTicket,
            messages: [PromptMessage(role: .user, content: "Write a long story about a lighthouse keeper.")],
            options: ReplyOptions(maxTokens: 400, temperature: 0.7, topP: 0.95, think: false),
            timeout: 120
        ) { event in
            guard case .text = event, cancelled.at == nil else { return }
            cancelled.at = Date()
            backend.cancel(longTicket)
        }
        let cancelToReturn = Self.interval(cancelled.at, long.returnedAt)
        let longPass = long.result?.cancelled == true && (cancelToReturn ?? .infinity) < 1.0

        // The next reply must not queue behind the cancelled one.
        let short = await runReply(
            ReplyTicket(),
            messages: [PromptMessage(role: .user, content: "Say hello in five words.")],
            options: ReplyOptions(maxTokens: 24, temperature: 0.7, topP: 0.95, think: false),
            timeout: 30
        )
        let firstEvent = Self.interval(short.calledAt, short.firstEventAt)
        let shortPass = short.result != nil && (firstEvent ?? .infinity) < 1.0

        // A ticket cancelled before its reply is even asked for.
        let earlyTicket = ReplyTicket()
        backend.cancel(earlyTicket)
        let early = await runReply(
            earlyTicket,
            messages: [PromptMessage(role: .user, content: "This reply was cancelled before it started.")],
            options: ReplyOptions(maxTokens: 24, temperature: 0.7, topP: 0.95, think: false),
            timeout: 10
        )
        let earlyReturn = Self.interval(early.calledAt, early.returnedAt)
        let earlyPass = early.result?.cancelled == true && (earlyReturn ?? .infinity) < 0.2

        record(name, pass: longPass && shortPass && earlyPass, [
            "long_cancel_to_return": Self.json(cancelToReturn),
            "long_cancelled": long.result.map { $0.cancelled as Any } ?? NSNull(),
            "long_partial_chars": long.result?.text.count ?? 0,
            "long_error": long.error ?? NSNull(),
            "long_pass": longPass,
            "next_first_event": Self.json(firstEvent),
            "next_error": short.error ?? NSNull(),
            "next_pass": shortPass,
            "precancelled_return": Self.json(earlyReturn),
            "precancelled_cancelled": early.result.map { $0.cancelled as Any } ?? NSNull(),
            "precancelled_pass": earlyPass,
        ])
    }

    // MARK: - e. no_false_bargein

    /// Leaves the microphone running for `externalOnset`.
    private func noFalseBargeIn(_ availability: VoiceInputAvailability) async {
        let name = "no_false_bargein"
        guard availability.isReady else {
            record(name, pass: nil, ["skipped": "microphone unavailable: \(availability)"])
            return
        }
        probe.reset()
        microphone.delegate = probe
        speech.delegate = probe
        do {
            try microphone.start(mode: .conversation)
        } catch {
            record(name, pass: false, ["error": error.localizedDescription])
            return
        }
        // Let the detector learn the room before anything plays.
        await pause(1.0)
        let quietOnsets = probe.onsets.count

        let start = Date()
        for sentence in Self.shortParagraph {
            speech.enqueue(sentence)
        }
        speech.finishTurn()
        let finished = await wait(upTo: 30) { !self.probe.speechFinishes.isEmpty }
        // The echo tail outlasts the last sample by a few hundred ms.
        await pause(0.5)
        let end = Date()

        let onsets = probe.onsets.filter { $0 >= start && $0 <= end }.count
        let audibleAt = probe.firstAudible(after: start)
        let spoken = probe.speechFinishes.first.map { $0.at.timeIntervalSince(start) }
        // Zero onsets proves nothing unless the microphone was delivering
        // audio the whole time; its level reports are that evidence.
        let microphoneLive = probe.inputLevels.contains { $0.at >= start }
        record(name, pass: finished && audibleAt != nil && microphoneLive && onsets == 0, [
            "onsets_while_speaking": onsets,
            "onsets_before_speaking": quietOnsets,
            "speech_seconds": Self.json(spoken),
            "audible": audibleAt != nil,
            "room": probe.inputDecibels(from: start.addingTimeInterval(-1), to: start),
            "residue_while_speaking": probe.inputDecibels(from: start, to: end),
            "note": "someone talking in the room during this check invalidates it",
        ])
    }

    // MARK: - f. external_onset

    /// Informational: a voice from outside the engine (a recording played
    /// with `AVAudioPlayer`, standing in for the user) while a reply is
    /// speaking. Reports whether it registered as a barge-in, how fast,
    /// and what the recogniser made of it.
    private func externalOnset(_ availability: VoiceInputAvailability) async {
        let name = "external_onset"
        defer { microphone.cancel() }
        guard availability.isReady, microphone.isListening else {
            record(name, pass: nil, ["skipped": "microphone not listening"])
            return
        }
        guard let url = Bundle.main.url(forResource: "sample-question-1", withExtension: "wav"),
              let player = try? AVAudioPlayer(contentsOf: url)
        else {
            record(name, pass: nil, ["skipped": "sample-question-1.wav is not in the bundle"])
            return
        }

        probe.reset()
        let speech = self.speech
        let start = Date()
        for sentence in Self.longParagraph {
            speech.enqueue(sentence)
        }
        speech.finishTurn()
        _ = await wait(upTo: 10) { self.probe.firstAudible(after: start) != nil }
        await pause(1.0)

        // What voice mode does on onset: cut the reply.
        probe.onOnset = { speech.stop() }
        player.prepareToPlay()
        let playAt = Date()
        player.play()
        let fired = await wait(upTo: 6) { self.probe.onsets.contains { $0 >= playAt } }
        let onsetAt = probe.onsets.first { $0 >= playAt }
        if fired {
            _ = await wait(upTo: 8) { self.probe.finals.contains { $0.at >= playAt } }
        }
        probe.onOnset = nil
        player.stop()
        speech.stop()

        record(name, pass: nil, [
            "onset_fired": fired,
            "onset_after_play": Self.seconds(playAt, onsetAt),
            "microphone_while_talking_over": probe.inputDecibels(from: playAt, to: onsetAt ?? Date()),
            "recording_seconds": Self.rounded(player.duration),
            "transcript": probe.finals.first { $0.at >= playAt }?.text ?? NSNull(),
            "speech_stopped_by_onset": probe.speechFinishes.contains { $0.interrupted && $0.at >= playAt },
        ])
    }

    // MARK: - Plumbing

    /// `prepare()` with a time limit: on a fresh install it waits on
    /// permission prompts.
    private func prepare(_ input: SpeechInput, timeout: TimeInterval = 30) async -> VoiceInputAvailability {
        let outcome = PrepareOutcome()
        Task { @MainActor in
            outcome.availability = await input.prepare()
        }
        guard await wait(upTo: timeout, until: { outcome.availability != nil }), let availability = outcome.availability else {
            return .unavailable("prepare() did not finish within \(Int(timeout)) s; is a permission prompt waiting?")
        }
        return availability
    }

    private func record(_ name: String, pass: Bool?, _ fields: [String: Any]) {
        verdicts.append((name, pass))
        var line = fields
        line["check"] = name
        line["pass"] = pass.map { $0 as Any } ?? NSNull()
        log.emit(line)
    }

    private func runReply(
        _ ticket: ReplyTicket,
        messages: [PromptMessage],
        options: ReplyOptions,
        timeout: TimeInterval,
        onEvent: @escaping (ReplyEvent) -> Void = { _ in }
    ) async -> ReplyOutcome {
        let outcome = ReplyOutcome()
        let backend = self.backend
        Task { @MainActor in
            outcome.calledAt = Date()
            do {
                outcome.result = try await backend.reply(
                    ticket,
                    session: Self.session,
                    messages: messages,
                    options: options
                ) { event in
                    // Stamped where it arrives, before the hop, so the time
                    // to first event is the engine's and not the main
                    // queue's.
                    let arrived = Date()
                    DispatchQueue.main.async {
                        if outcome.firstEventAt == nil {
                            outcome.firstEventAt = arrived
                        }
                        onEvent(event)
                    }
                }
            } catch {
                outcome.error = error.localizedDescription
            }
            outcome.returnedAt = Date()
            outcome.done = true
        }
        _ = await wait(upTo: timeout) { outcome.done }
        return outcome
    }

    private func wait(upTo seconds: TimeInterval, until condition: () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() {
            if Date() >= deadline { return false }
            await pause(0.01)
        }
        return true
    }

    private func pause(_ seconds: TimeInterval) async {
        guard seconds > 0 else { return }
        try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
    }

    private static func rounded(_ value: TimeInterval) -> Double {
        (value * 1000).rounded() / 1000
    }

    /// Levels to a tenth of a decibel.
    private static func tenths(_ decibels: Double) -> Double {
        (decibels * 10).rounded() / 10
    }

    private static func interval(_ from: Date?, _ to: Date?) -> TimeInterval? {
        guard let from, let to else { return nil }
        return to.timeIntervalSince(from)
    }

    private static func seconds(_ from: Date, _ to: Date?) -> Any {
        json(to.map { $0.timeIntervalSince(from) })
    }

    private static func json(_ value: TimeInterval?) -> Any {
        guard let value, value.isFinite else { return NSNull() }
        return rounded(value)
    }
}

private final class PrepareOutcome {
    var availability: VoiceInputAvailability?
}

private final class WarmUpOutcome {
    var message: String?
    var done = false
}

private final class ReplyOutcome {
    var calledAt: Date?
    var firstEventAt: Date?
    var returnedAt: Date?
    var result: ReplyResult?
    var error: String?
    var done = false
}

private final class Stamp {
    var at: Date?
}

/// Every callback the inputs and the synthesizer make, with when it came.
private final class Probe: SpeechInputDelegate, SpeechOutputDelegate {
    var onsets: [Date] = []
    var finals: [(at: Date, text: String)] = []
    var failures: [String] = []
    var availability: [String] = []
    var speechStarts: [Date] = []
    var speechFinishes: [(at: Date, interrupted: Bool)] = []
    var outputLevels: [(at: Date, level: Float)] = []
    var inputLevels: [(at: Date, level: Float)] = []
    /// Runs on every onset, the way voice mode reacts to a barge-in.
    var onOnset: (() -> Void)?

    func reset() {
        onsets.removeAll()
        finals.removeAll()
        failures.removeAll()
        availability.removeAll()
        speechStarts.removeAll()
        speechFinishes.removeAll()
        outputLevels.removeAll()
        inputLevels.removeAll()
        onOnset = nil
    }

    func firstAudible(after start: Date) -> Date? {
        outputLevels.first { $0.at >= start && $0.level > 0.02 }?.at
    }

    /// The input level reports between two times, in dBFS (the inverse of
    /// `AudioLevel.normalized`, so anything at or below -50 dBFS reads as
    /// -50).
    func decibels(from start: Date, to end: Date) -> [(at: Date, decibels: Double)] {
        inputLevels
            .filter { $0.at >= start && $0.at <= end }
            .map { ($0.at, Double($0.level) * 50 - 50) }
    }

    func medianDecibels(from start: Date, to end: Date) -> Double? {
        let sorted = decibels(from: start, to: end).map(\.decibels).sorted()
        guard !sorted.isEmpty else { return nil }
        return sorted[sorted.count / 2]
    }

    func speechInputDidChangeAvailability(_ availability: VoiceInputAvailability) {
        self.availability.append("\(availability)")
    }

    func speechInputDidUpdatePartial(_ text: String) {}

    func speechInputDidFinalize(_ text: String) {
        finals.append((Date(), text))
    }

    func speechInputDidUpdateLevel(_ level: Float) {
        inputLevels.append((Date(), level))
    }

    /// The echo-cancelled microphone's level between two times, in dBFS
    /// (the inverse of `AudioLevel.normalized`, so anything at or below
    /// -50 dBFS reads as -50): what the voice-activity detector was
    /// looking at, for tuning its thresholds from real numbers.
    func inputDecibels(from start: Date, to end: Date) -> [String: Any] {
        let decibels = self.decibels(from: start, to: end).map(\.decibels).sorted()
        guard !decibels.isEmpty else { return ["reports": 0] }
        func percentile(_ p: Double) -> Double {
            let value = decibels[min(decibels.count - 1, Int(Double(decibels.count) * p))]
            return (value * 10).rounded() / 10
        }
        return [
            "reports": decibels.count,
            "p50_dbfs": percentile(0.5),
            "p90_dbfs": percentile(0.9),
            "max_dbfs": percentile(1),
        ]
    }

    func speechInputDidDetectSpeechOnset() {
        onsets.append(Date())
        onOnset?()
    }

    func speechInputDidFail(_ error: Error) {
        failures.append(error.localizedDescription)
    }

    func speechOutputDidStart() {
        speechStarts.append(Date())
    }

    func speechOutputDidFinish(interrupted: Bool) {
        speechFinishes.append((Date(), interrupted))
    }

    func speechOutputLevel(_ level: Float) {
        outputLevels.append((Date(), level))
    }
}

/// What the hub's two playback channels are doing, every 10 ms: whether
/// audio is queued and the level of what is leaving the speaker now, read
/// from the channels themselves rather than from the sample input's or
/// the synthesizer's own state and meters. Main thread.
private final class ChannelRecorder {
    struct Sample {
        let at: Date
        let cueActive: Bool
        let cueLevel: Float
        let speechActive: Bool
        let speechLevel: Float

        var cueSounding: Bool { cueActive || cueLevel > 0 }
        var speechSounding: Bool { speechActive || speechLevel > 0 }
    }

    private(set) var samples: [Sample] = []
    private var timer: Timer?

    func start() {
        stop()
        samples.removeAll()
        let hub = AudioEngineHub.shared
        let timer = Timer(timeInterval: 0.01, repeats: true) { [weak self] _ in
            self?.samples.append(Sample(
                at: Date(),
                cueActive: hub.cue.isActive,
                cueLevel: hub.cue.level,
                speechActive: hub.speech.isActive,
                speechLevel: hub.speech.level
            ))
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }
}

/// Writes each result as one JSON line to stdout and to
/// Documents/audio-check.jsonl.
private final class CheckLog {
    private let url = FileManager.default
        .urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("audio-check.jsonl")

    func emit(_ object: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let line = String(data: data, encoding: .utf8)
        else {
            print("PIECHECK {\"check\":\"log\",\"pass\":null,\"error\":\"unserialisable result\"}")
            return
        }
        print("PIECHECK \(line)")
        fflush(stdout)

        let bytes = Data((line + "\n").utf8)
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: bytes)
        } else {
            try? bytes.write(to: url)
        }
    }
}
