import AVFoundation
import Foundation

/// `SpeechOutput` rendered through the app's shared audio engine.
///
/// `AVSpeechSynthesizer.speak` plays through an audio path of its own that
/// the echo canceller never sees, so the microphone heard the reply as if
/// someone were talking and voice mode could not listen while it spoke.
/// Here the synthesizer only renders (`write(_:toBufferCallback:)`), one
/// sentence at a time and in order, and the buffers play on the hub's
/// speech channel: the same engine whose input node cancels them out.
///
/// Call from the main thread. Delegate calls arrive on the main queue,
/// asynchronously and in order, addressed to the delegate that was set
/// when the event happened: a turn cut off by `stop()` reports its end to
/// whoever owned it, even if a new owner has already taken over.
final class SpeechSynthesis: NSObject, SpeechOutput {

    weak var delegate: SpeechOutputDelegate?
    /// True from the first `enqueue` of a turn until every rendered buffer
    /// has been heard and `finishTurn()` was called, or until `stop()`.
    private(set) var isSpeaking = false
    var voiceIdentifier: String?
    var rate: Float = 0.5
    /// About -14 dB: still audible, so a false alarm costs nothing, but
    /// well under a voice talking over it.
    var isDucked = false {
        didSet { hub.speech.gain = isDucked ? Self.duckedGain : 1 }
    }
    private static let duckedGain: Float = 0.2

    private let synthesizer = AVSpeechSynthesizer()
    private let hub = AudioEngineHub.shared

    /// Names the current turn. Every deferred callback (rendered audio,
    /// played-back completions, the meter) carries the turn it belongs to
    /// and is dropped once that turn is over, so audio from a stopped reply
    /// can never play later.
    private var turn = 0
    private var pending: [String] = []
    /// The sentence the synthesizer is rendering now. One at a time, so
    /// the buffers arrive in speaking order.
    private var rendering: AVSpeechUtterance?
    private var renderingProducedAudio = false
    private var unplayedBuffers = 0
    private var turnIsOpen = false
    private var turnVoice: AVSpeechSynthesisVoice?
    /// Whether `turnVoice` is the one chosen in Settings, and whether it
    /// has played anything yet this turn; see `fallBackIfChosenVoiceIsSilent`.
    private var turnVoiceIsChosen = false
    private var turnVoiceHasSpoken = false
    private var meter: Timer?
    private var meterLevel: Float = 0

    override init() {
        super.init()
        synthesizer.delegate = self
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(audioWasInterrupted(_:)),
            name: AudioEngineHub.audioWasInterrupted,
            object: nil
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - SpeechOutput

    func enqueue(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if !isSpeaking {
            guard beginTurn() else { return }
        }
        // Text arriving after `finishTurn()` but before the queue drained
        // continues the same turn, so there is one start and one finish.
        turnIsOpen = true
        pending.append(trimmed)
        renderNextIfIdle()
    }

    func finishTurn() {
        turnIsOpen = false
        // If everything has already been heard, the last completion has
        // been and gone; settle here rather than wait for one that will
        // never come.
        settleIfDone()
    }

    func stop() {
        guard isSpeaking else { return }
        endTurn(interrupted: true)
    }

    /// Installed voices in the current language (or, with none fit to
    /// choose there, its language family), in the order automatic
    /// selection prefers them (see `automaticVoices`). Legacy voices come
    /// last and are marked (`SpeechVoiceInfo.isLegacy`) so the picker can
    /// hide or mark them. Novelty voices are not listed, and neither is
    /// Personal Voice, which has a list of its own (`personalVoices()`).
    static func availableVoices() -> [SpeechVoiceInfo] {
        let language = AVSpeechSynthesisVoice.currentLanguageCode()
        let pool = voicePool(language: language)
        let modern = ranked(pool.filter { !isLegacyVoice($0) }, language: language)
        let legacy = pool.filter(isLegacyVoice).sorted { ($0.name, $0.language) < ($1.name, $1.language) }
        return (modern + legacy).map(info)
    }

    // MARK: - Personal Voice

    /// Whether the app may speak in the user's Personal Voice. Reading it
    /// never asks the user anything.
    static var personalVoiceAccess: PersonalVoiceAccess {
        access(AVSpeechSynthesizer.personalVoiceAuthorizationStatus)
    }

    /// Asks the user to let the app speak in their Personal Voice. For the
    /// settings screen's "Use my Personal Voice" only: nothing else calls
    /// it, so the system prompt appears only when the user asked for it.
    /// The system asks once; after that this returns the answer given,
    /// which the user can change only in the Settings app.
    static func requestPersonalVoiceAccess() async -> PersonalVoiceAccess {
        access(await AVSpeechSynthesizer.requestPersonalVoiceAuthorization())
    }

    /// The user's Personal Voices, once the app may use them, and empty
    /// otherwise; never asks. A voice the user allows or finishes creating
    /// while the app runs can take a moment to be listed:
    /// `AVSpeechSynthesizer.availableVoicesDidChangeNotification` says when.
    /// Choosing one is choosing it in `voiceIdentifier`; automatic
    /// selection never picks it, since it is the user's own voice.
    static func personalVoices() -> [SpeechVoiceInfo] {
        guard AVSpeechSynthesizer.personalVoiceAuthorizationStatus == .authorized else { return [] }
        return AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.voiceTraits.contains(.isPersonalVoice) }
            .sorted { ($0.name, $0.identifier) < ($1.name, $1.identifier) }
            .map(info)
    }

    private static func access(_ status: AVSpeechSynthesizer.PersonalVoiceAuthorizationStatus) -> PersonalVoiceAccess {
        switch status {
        case .authorized: return .authorized
        case .denied: return .denied
        case .unsupported: return .unsupported
        case .notDetermined: return .notDetermined
        @unknown default: return .unsupported
        }
    }

    // MARK: - Turn

    private func beginTurn() -> Bool {
        turn += 1
        isDucked = false
        let target = delegate
        do {
            try hub.beginPlayback(.speech)
        } catch {
            // Reported as a turn that was cut off at once, so whoever is
            // waiting for the end of speech moves on. Nothing is latched:
            // the next chunk tries the engine again, so a failure while a
            // call holds the audio does not silence the app afterwards.
            print("[audio] speech could not start: \(error.localizedDescription)")
            deliver {
                target?.speechOutputDidStart()
                target?.speechOutputDidFinish(interrupted: true)
            }
            return false
        }
        isSpeaking = true
        let resolved = resolveVoice()
        turnVoice = resolved.voice
        turnVoiceIsChosen = resolved.chosen
        turnVoiceHasSpoken = false
        Self.logVoice(resolved.voice, how: resolved.how)
        startMeter()
        deliver { target?.speechOutputDidStart() }
        return true
    }

    private func settleIfDone() {
        guard isSpeaking, !turnIsOpen, rendering == nil, pending.isEmpty, unplayedBuffers == 0 else { return }
        endTurn(interrupted: false)
    }

    private func endTurn(interrupted: Bool) {
        turn += 1
        if rendering != nil {
            synthesizer.stopSpeaking(at: .immediate)
        }
        hub.speech.stop()
        pending.removeAll()
        rendering = nil
        unplayedBuffers = 0
        turnIsOpen = false
        isSpeaking = false
        meter?.invalidate()
        meter = nil
        meterLevel = 0

        let target = delegate
        deliver {
            target?.speechOutputLevel(0)
            target?.speechOutputDidFinish(interrupted: interrupted)
        }
        hub.endPlayback(.speech)
    }

    // MARK: - Rendering

    private func renderNextIfIdle() {
        guard rendering == nil, !pending.isEmpty else { return }
        let utterance = AVSpeechUtterance(string: pending.removeFirst())
        utterance.voice = turnVoice
        utterance.rate = min(max(rate, AVSpeechUtteranceMinimumSpeechRate), AVSpeechUtteranceMaximumSpeechRate)
        rendering = utterance
        renderingProducedAudio = false

        let turn = self.turn
        // One converter per sentence: it carries resampler state from one
        // buffer to the next, and its tail is flushed when the sentence
        // ends. The coalescer turns the synthesizer's 11 ms pieces into
        // 100 ms buffers before they cost a hop to the main queue each.
        let converter = PCMConverter(outputFormat: AudioEngineHub.playbackFormat)
        let coalescer = BufferCoalescer(format: AudioEngineHub.playbackFormat, chunkFrames: 4800)
        synthesizer.write(utterance) { [weak self] buffer in
            guard let pcm = buffer as? AVAudioPCMBuffer else { return }
            if pcm.frameLength == 0 {
                // The zero-length buffer marks the end of the sentence. It
                // comes through this same callback after the sentence's
                // audio, so it is handled after that audio is queued. The
                // synthesizer has been seen to send it twice; the second
                // finds nothing left to flush and a sentence already over.
                let tail = coalescer.append(converter.finish()) + coalescer.flush()
                DispatchQueue.main.async {
                    self?.play(tail, of: utterance, turn: turn)
                    self?.renderingDidEnd(utterance, turn: turn)
                }
            } else {
                let ready = coalescer.append(converter.convert(pcm))
                guard !ready.isEmpty else { return }
                DispatchQueue.main.async {
                    self?.play(ready, of: utterance, turn: turn)
                }
            }
        }
        // A sentence the synthesizer renders nothing for (a voice that
        // failed to load) must not leave the turn speaking forever. The
        // render is stopped too: the synthesizer queues what it is given,
        // so a render that never ends would hold up every sentence after it.
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
            guard let self, self.rendering === utterance, !self.renderingProducedAudio else { return }
            print("[audio] the synthesizer rendered nothing for a sentence in 5 s; stopping it")
            self.synthesizer.stopSpeaking(at: .immediate)
            self.renderingDidEnd(utterance, turn: turn)
        }
    }

    /// Queues `buffers` of `utterance`. Audio of a sentence that is no
    /// longer the one rendering is dropped: one given up on above, or
    /// handed to another voice, must not play late, out of order or twice.
    private func play(_ buffers: [AVAudioPCMBuffer], of utterance: AVSpeechUtterance, turn: Int) {
        guard turn == self.turn, isSpeaking, rendering === utterance else { return }
        if !buffers.isEmpty {
            renderingProducedAudio = true
            turnVoiceHasSpoken = true
        }
        for buffer in buffers {
            unplayedBuffers += 1
            hub.speech.schedule(buffer) { [weak self] in
                self?.bufferWasHeard(turn: turn)
            }
        }
    }

    private func bufferWasHeard(turn: Int) {
        guard turn == self.turn else { return }
        unplayedBuffers = max(0, unplayedBuffers - 1)
        settleIfDone()
    }

    private func renderingDidEnd(_ utterance: AVSpeechUtterance, turn: Int) {
        guard turn == self.turn, rendering === utterance else { return }
        rendering = nil
        if !renderingProducedAudio {
            fallBackIfChosenVoiceIsSilent(lost: utterance)
        }
        renderNextIfIdle()
        settleIfDone()
    }

    /// A voice chosen in Settings that renders nothing through
    /// `write(_:toBufferCallback:)` (a Personal Voice the synthesizer will
    /// not hand over as audio, a voice that failed to load) would leave
    /// the whole reply silent, sentence after sentence. The rest of the
    /// turn moves to the automatic voice, starting again with the sentence
    /// that was lost. Only while the chosen voice has not played anything
    /// this turn, and for a sentence with a letter or digit in it, so a
    /// voice that works is never dropped over a line of punctuation.
    private func fallBackIfChosenVoiceIsSilent(lost utterance: AVSpeechUtterance) {
        let text = utterance.speechString
        guard turnVoiceIsChosen, !turnVoiceHasSpoken, text.contains(where: { $0.isLetter || $0.isNumber }) else {
            return
        }
        let language = AVSpeechSynthesisVoice.currentLanguageCode()
        let fallback = Self.automaticVoices(language: language).first ?? AVSpeechSynthesisVoice(language: language)
        turnVoiceIsChosen = false
        guard let fallback, fallback.identifier != turnVoice?.identifier else { return }
        print("[audio] the voice chosen in Settings (\(turnVoice?.identifier ?? "none")) rendered nothing; "
            + "this reply goes on in the automatic voice")
        turnVoice = fallback
        Self.logVoice(fallback, how: "automatic, because the voice chosen in Settings rendered nothing")
        pending.insert(text, at: 0)
    }

    // MARK: - Meter

    private func startMeter() {
        meter?.invalidate()
        meterLevel = 0
        let turn = self.turn
        let timer = Timer(timeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            guard let self, self.turn == turn else { return }
            let level = self.hub.speech.level
            // Rise at once, fall over a few frames, the way a VU needle
            // does; the raw windows flicker between syllables.
            self.meterLevel = level >= self.meterLevel ? level : self.meterLevel * 0.6 + level * 0.4
            self.delegate?.speechOutputLevel(self.meterLevel)
        }
        // Common modes, so the meter keeps moving while a list scrolls.
        RunLoop.main.add(timer, forMode: .common)
        meter = timer
    }

    // MARK: - Voices

    /// The voice a turn speaks with, and how it came to be chosen, for the
    /// log. A voice chosen in Settings is used as it is, legacy or not; a
    /// Personal Voice only while the app is still allowed to use it, since
    /// without that it renders nothing.
    private func resolveVoice() -> (voice: AVSpeechSynthesisVoice?, how: String, chosen: Bool) {
        var how = "automatic"
        if let voiceIdentifier {
            if let voice = AVSpeechSynthesisVoice(identifier: voiceIdentifier),
               !voice.voiceTraits.contains(.isPersonalVoice)
                || AVSpeechSynthesizer.personalVoiceAuthorizationStatus == .authorized {
                return (voice, "chosen in Settings", true)
            }
            how = "automatic, because the voice chosen in Settings (\(voiceIdentifier)) is not available"
        }
        let language = AVSpeechSynthesisVoice.currentLanguageCode()
        let voice = Self.automaticVoices(language: language).first
            ?? AVSpeechSynthesisVoice(language: language)
        return (voice, how, false)
    }

    /// Automatic selection, best first: Premium, then Enhanced, then the
    /// system's own default voice for the language (what Spoken Content
    /// speaks with, `AVSpeechSynthesisVoice(language:)`), then the other
    /// Default voices, then the Eloquence voices. Never a legacy or novelty
    /// voice, and never Personal Voice.
    ///
    /// Sorting the Default voices by name alone, as this used to, put
    /// "Eddy" first in US English (listed on a Mac, which ships the same
    /// voices as iOS 17 and later): an Eloquence voice, the formant
    /// synthesizer screen-reader users know, clear at speed but plainly
    /// synthetic, then Flo, another, then Fred, a MacinTalk voice from the
    /// 1980s. With no Premium or Enhanced voice downloaded, the user heard
    /// Eddy instead of Samantha, the system's own voice.
    private static func automaticVoices(language: String) -> [AVSpeechSynthesisVoice] {
        ranked(voicePool(language: language).filter { !isLegacyVoice($0) }, language: language)
    }

    /// The installed voices of `language`, or, when none of them is fit to
    /// choose automatically, of its language family (en-GB, en-AU and the
    /// rest for en-US), so a regional setting with no voice of its own
    /// still gets a natural one. Novelty voices and Personal Voice are
    /// left out.
    private static func voicePool(language: String) -> [AVSpeechSynthesisVoice] {
        let installed = AVSpeechSynthesisVoice.speechVoices().filter { voice in
            !voice.voiceTraits.contains(.isNoveltyVoice) && !voice.voiceTraits.contains(.isPersonalVoice)
        }
        let exact = installed.filter { $0.language == language }
        if exact.contains(where: { !isLegacyVoice($0) }) {
            return exact
        }
        let family = languageFamily(language)
        return installed.filter { languageFamily($0.language) == family }
    }

    private static func ranked(_ voices: [AVSpeechSynthesisVoice], language: String) -> [AVSpeechSynthesisVoice] {
        let systemDefault = AVSpeechSynthesisVoice(language: language)?.identifier
        return voices.sorted { lhs, rhs in
            let left = (tier(lhs), lhs.identifier == systemDefault ? 1 : 0)
            let right = (tier(rhs), rhs.identifier == systemDefault ? 1 : 0)
            if left != right { return left > right }
            // Within a tier, the language's own voices before its regional
            // cousins, then by name, so the choice is the same on every run.
            let leftExact = lhs.language == language ? 0 : 1
            let rightExact = rhs.language == language ? 0 : 1
            return (leftExact, lhs.name, lhs.identifier) < (rightExact, rhs.name, rhs.identifier)
        }
    }

    private static func tier(_ voice: AVSpeechSynthesisVoice) -> Int {
        switch voice.quality {
        case .premium: return 4
        case .enhanced: return 3
        default: return isEloquenceVoice(voice) ? 1 : 2
        }
    }

    private static func languageFamily(_ language: String) -> String {
        String(language.prefix { $0 != "-" && $0 != "_" }).lowercased()
    }

    /// A voice that is never chosen automatically: a novelty voice, or one
    /// of the MacinTalk voices of the 1980s and 90s that Apple still ships
    /// (Fred, Junior, Kathy, Ralph and the rest). Only some of them carry
    /// `isNoveltyVoice` (Fred, Junior, Kathy and Ralph do not), so they are
    /// also matched by name, against both the voice's display name and the
    /// last component of its identifier, because the two disagree: the
    /// voice shown as "Jester" is com.apple.speech.synthesis.voice.Hysterical,
    /// "Superstar" is Princess and "Wobble" is Deranged. The list also
    /// covers the older names macOS used (Agnes, Bruce, Vicki, Victoria).
    ///
    /// Names are matched only under the MacinTalk identifiers
    /// (`macinTalkPrefix`, which all of them use), so a Personal Voice the
    /// user named "Kathy", or a later voice that reuses one of these names,
    /// is not taken for one.
    static func isLegacyVoice(_ voice: AVSpeechSynthesisVoice) -> Bool {
        if voice.voiceTraits.contains(.isNoveltyVoice) {
            return true
        }
        guard voice.identifier.hasPrefix(macinTalkPrefix) else { return false }
        let lastComponent = voice.identifier.split(separator: ".").last.map(String.init) ?? ""
        return [voice.name, lastComponent].contains { legacyNames.contains(normalizedName($0)) }
    }

    private static let macinTalkPrefix = "com.apple.speech.synthesis.voice."

    private static let legacyNames: Set<String> = [
        "agnes", "albert", "badnews", "bahh", "bells", "boing", "bruce", "bubbles",
        "cellos", "deranged", "fred", "goodnews", "hysterical", "jester", "junior",
        "kathy", "organ", "pipeorgan", "princess", "ralph", "superstar", "trinoids",
        "vicki", "victoria", "whisper", "wobble", "zarvox",
    ]

    private static func normalizedName(_ name: String) -> String {
        name.lowercased().filter { $0.isLetter }
    }

    /// The Eloquence voices (Eddy, Flo, Grandma, Grandpa, Reed, Rocko,
    /// Sandy, Shelley): a choice for those who want them, ranked below
    /// every other Default voice for everyone else.
    private static func isEloquenceVoice(_ voice: AVSpeechSynthesisVoice) -> Bool {
        voice.identifier.hasPrefix("com.apple.eloquence.")
    }

    private static func info(_ voice: AVSpeechSynthesisVoice) -> SpeechVoiceInfo {
        SpeechVoiceInfo(
            id: voice.identifier,
            name: voice.name,
            quality: qualityName(voice.quality),
            language: voice.language,
            isLegacy: isLegacyVoice(voice),
            isPersonalVoice: voice.voiceTraits.contains(.isPersonalVoice)
        )
    }

    private static func qualityName(_ quality: AVSpeechSynthesisVoiceQuality) -> String {
        switch quality {
        case .premium: return "Premium"
        case .enhanced: return "Enhanced"
        default: return "Default"
        }
    }

    /// What the log last said this process speaks with. The first turn of
    /// a process names its voice, and a turn names it again only when it
    /// changed, so the console says what the user is hearing.
    private static var loggedVoice: String?

    private static func logVoice(_ voice: AVSpeechSynthesisVoice?, how: String) {
        let key = "\(voice?.identifier ?? "") \(how)"
        guard key != loggedVoice else { return }
        loggedVoice = key
        let language = AVSpeechSynthesisVoice.currentLanguageCode()
        if let voice {
            print("[audio] voice \(voice.name) (\(qualityName(voice.quality)), \(voice.identifier)), \(how)")
        } else {
            print("[audio] voice: none installed for \(language); the synthesizer's own default, \(how)")
        }
        print("[audio] \(voiceCensus(language: language))")
    }

    /// One line: the voices installed for `language` by quality, how many
    /// of the Default ones are legacy or Eloquence, and the system's own
    /// default voice for it.
    private static func voiceCensus(language: String) -> String {
        let voices = AVSpeechSynthesisVoice.speechVoices().filter { $0.language == language }
        let personal = voices.filter { $0.voiceTraits.contains(.isPersonalVoice) }.count
        let installed = voices.filter { !$0.voiceTraits.contains(.isPersonalVoice) }
        let premium = installed.filter { $0.quality == .premium }.count
        let enhanced = installed.filter { $0.quality == .enhanced }.count
        let standard = installed.count - premium - enhanced
        let legacy = installed.filter(isLegacyVoice).count
        let eloquence = installed.filter(isEloquenceVoice).count
        let systemDefault = AVSpeechSynthesisVoice(language: language)
            .map { "\($0.name) (\($0.identifier))" } ?? "none"
        return "voices installed for \(language): \(premium) Premium, \(enhanced) Enhanced, "
            + "\(standard) Default (\(legacy) legacy or novelty, \(eloquence) Eloquence), "
            + "\(personal) Personal Voice; system default \(systemDefault)"
    }

    // MARK: - Plumbing

    private func deliver(_ work: @escaping () -> Void) {
        DispatchQueue.main.async(execute: work)
    }

    @objc private func audioWasInterrupted(_ note: Notification) {
        guard isSpeaking else { return }
        endTurn(interrupted: true)
    }
}

extension SpeechSynthesis: AVSpeechSynthesizerDelegate {

    /// The zero-length buffer is what normally ends a sentence. Should it
    /// ever not arrive, the queue must not stall on that sentence, so the
    /// synthesizer's own end-of-utterance callback is a backstop, delayed
    /// so any audio still in flight from the render callback lands first.
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            guard let self, self.rendering === utterance else { return }
            self.renderingDidEnd(utterance, turn: self.turn)
        }
    }
}
