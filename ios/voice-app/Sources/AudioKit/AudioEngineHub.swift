import AVFoundation
import AudioToolbox
import Foundation

/// The app's one `AVAudioEngine` and the `AVAudioSession` it runs in.
///
/// Speech output and microphone capture share this engine because that is
/// what makes echo cancellation possible: with voice processing on, the
/// input node subtracts whatever the engine's own output node is playing
/// from what the microphone hears. A reply played any other way
/// (`AVSpeechSynthesizer.speak`, a second engine, `AVAudioPlayer`) reaches
/// the microphone untouched and is indistinguishable from the user
/// talking, which is why talking over a reply could never work before.
///
/// The engine is rebuilt, never reconfigured, when its input side has to
/// change (no input, raw microphone, voice-processed microphone). Voice
/// processing can only be toggled on a stopped engine, it changes the
/// input node's format underneath existing connections, and a fresh
/// engine is the dependable way to get a consistent graph. Audio still
/// queued on the playback channels is carried over to the new engine, so
/// a rebuild in the middle of a reply costs a short hiccup, not the rest
/// of the reply.
///
/// Audio session, chosen once per engine build (the category never
/// changes, because switching categories mid-conversation is an audible
/// gap and re-routes the audio):
///   - category `.playAndRecord` for the app's lifetime, which also
///     ignores the ring/silent switch, so a reply is audible either way;
///   - `.defaultToSpeaker`, so play-and-record does not send replies to
///     the earpiece, re-checked after every route change;
///   - mode `.voiceChat` while the voice-processed microphone is in use:
///     it is the mode the voice-processing unit is designed for, and iOS
///     switches to it implicitly anyway, so setting it ourselves keeps the
///     route decision ours;
///   - mode `.spokenAudio` otherwise (read-aloud, dictation), the mode the
///     previous build shipped with;
///   - Bluetooth A2DP always, and HFP only while a microphone is open
///     (dictation, voice mode), so AirPods can carry the user's voice then
///     while read-aloud on its own stays on full-quality A2DP.
/// The session is active only while someone uses the engine: a second
/// after the last client leaves it is deactivated, telling other apps, so
/// music the app paused resumes and the voice-chat route is let go.
///
/// Main thread only, except `isEchoLikely`, which the input tap reads.
final class AudioEngineHub {

    static let shared = AudioEngineHub()

    /// What the input side of the engine is built for.
    enum InputConfiguration: String {
        /// Playback only. The input node is never touched, so reading a
        /// reply aloud does not open the microphone or show the
        /// recording indicator.
        case none
        /// The raw microphone, for dictation.
        case plain
        /// The echo-cancelled microphone, for voice mode.
        case voiceProcessed
    }

    /// Who is using the engine. It runs while anyone is, and stops (which
    /// closes the microphone) when nobody is.
    enum Client: Hashable {
        case speech
        case cue
        case input
    }

    /// Posted on the main queue when the system took the audio away (a
    /// call, Siri, an alarm), audio services restarted, or the engine
    /// could not be brought back after a hardware change. Whatever was
    /// playing or listening has stopped; clients end their turns and
    /// sessions when they receive it.
    static let audioWasInterrupted = Notification.Name("AudioEngineHub.audioWasInterrupted")

    /// Posted on the main queue when the microphone's input route went away
    /// during a hardware change and did not come back. Only the microphone
    /// has stopped; playback carries on.
    static let inputWasLost = Notification.Name("AudioEngineHub.inputWasLost")

    /// Every channel plays this format. Fixed, so connections never change
    /// when a different voice or recording comes along; the main mixer
    /// converts it to whatever the hardware runs at.
    static let playbackFormat = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!

    /// Voice processing plays the reply at a noticeably lower level than
    /// the same audio without it on the iPhone speaker (the voice-chat
    /// volume curve plus the processing's own output stage). This much
    /// make-up gain, applied ahead of a peak limiter so the louder speech
    /// cannot clip, brings replies back to a normal speaking level.
    private static let voiceProcessingMakeUpGain: Float = 6

    /// How long the engine has to have had no clients before the session is
    /// deactivated. Long enough that one turn handing over to the next
    /// (the sample question to the microphone, one reply to the next) does
    /// not deactivate and reactivate the session in between.
    private static let deactivationDelay: TimeInterval = 1.0
    /// A route change can leave the session with no input for a moment
    /// (AirPods connecting or disconnecting hand the microphone over). The
    /// microphone is tried again this often, this many times, before its
    /// user is told it has gone.
    private static let inputRetryDelay: TimeInterval = 0.4
    private static let inputRetryLimit = 3

    private static let peakLimiter = AudioComponentDescription(
        componentType: kAudioUnitType_Effect,
        componentSubType: kAudioUnitSubType_PeakLimiter,
        componentManufacturer: kAudioUnitManufacturer_Apple,
        componentFlags: 0,
        componentFlagsMask: 0
    )

    /// The spoken reply.
    let speech = PlaybackChannel()
    /// The bundled sample question, played as if the user were asking it.
    let cue = PlaybackChannel()

    private var engine = AVAudioEngine()
    /// What the current engine was built for; nil until the first build,
    /// and after anything that makes the current engine unusable.
    private var built: InputConfiguration?
    /// Voice processing was asked for and the input node refused it.
    private var echoCancellationFailed = false
    private var tapBlock: AVAudioNodeTapBlock?
    private var tapInstalled = false
    /// The format the input tap was installed with and the hardware
    /// output format the engine was built against, so a configuration
    /// change can tell whether either actually moved.
    private var tapFormat: AVAudioFormat?
    private var builtOutputFormat: AVAudioFormat?
    private var clients: Set<Client> = []
    private var sessionActive = false
    /// After a media-services reset every audio object is an orphan;
    /// calling into the old engine is not safe.
    private var engineIsOrphaned = false
    private var configurationObserver: NSObjectProtocol?
    /// When recent configuration-change rebuilds happened; see
    /// `engineConfigurationChanged`.
    private var recentRebuilds: [Date] = []
    private var pendingDeactivation: DispatchWorkItem?
    private var pendingInputRetry: DispatchWorkItem?
    private let echo = EchoGate()

    private init() {
        for channel in [speech, cue] {
            let source = ObjectIdentifier(channel)
            channel.activityChanged = { [echo] active in
                echo.set(source, active: active)
            }
        }
        let center = NotificationCenter.default
        let session = AVAudioSession.sharedInstance()
        center.addObserver(
            self,
            selector: #selector(sessionInterrupted(_:)),
            name: AVAudioSession.interruptionNotification,
            object: session
        )
        center.addObserver(
            self,
            selector: #selector(mediaServicesWereReset(_:)),
            name: AVAudioSession.mediaServicesWereResetNotification,
            object: session
        )
        center.addObserver(
            self,
            selector: #selector(routeChanged(_:)),
            name: AVAudioSession.routeChangeNotification,
            object: session
        )
    }

    /// Whether the speaker is playing (or was a moment ago) something the
    /// microphone may still hear a residue of. Thread-safe; the voice
    /// activity detector reads it on the tap thread.
    var isEchoLikely: Bool { echo.isEchoLikely }

    // MARK: - Playback

    /// Claims the engine for one of the playback channels and starts it.
    func beginPlayback(_ client: Client) throws {
        let joining = !clients.subtracting([client]).isEmpty
        claim(client)
        do {
            // Nobody else is using the engine, so if it was last built
            // with a microphone this is the moment to close it again.
            if built == nil || (!joining && built != InputConfiguration.none) {
                try rebuild(.none)
            }
            try startIfNeeded()
        } catch {
            clients.remove(client)
            stopIfIdle()
            throw error
        }
    }

    func endPlayback(_ client: Client) {
        guard clients.remove(client) != nil else { return }
        stopIfIdle()
    }

    // MARK: - Input

    /// Opens the microphone with `configuration` and delivers its buffers
    /// to `tap` on the engine's tap thread. Rebuilds the engine when it was
    /// built for something else; playback in progress carries over.
    @discardableResult
    func startInput(
        _ configuration: InputConfiguration,
        tap: @escaping AVAudioNodeTapBlock
    ) throws -> AVAudioFormat {
        stopInput()
        claim(.input)
        do {
            if built != configuration {
                try rebuild(configuration)
            }
            tapBlock = tap
            let format = try installTap()
            try startIfNeeded()
            return format
        } catch {
            removeTap()
            tapBlock = nil
            clients.remove(.input)
            if clients.isEmpty {
                stopIfIdle()
            } else {
                try? startIfNeeded()
            }
            throw error
        }
    }

    func stopInput() {
        guard clients.contains(.input) else { return }
        pendingInputRetry?.cancel()
        pendingInputRetry = nil
        removeTap()
        tapBlock = nil
        clients.remove(.input)
        stopIfIdle()
    }

    // MARK: - Engine lifecycle

    private func claim(_ client: Client) {
        clients.insert(client)
        pendingDeactivation?.cancel()
        pendingDeactivation = nil
    }

    private func stopIfIdle() {
        guard clients.isEmpty else { return }
        if !engineIsOrphaned, engine.isRunning {
            engine.stop()
        }
        scheduleDeactivation()
    }

    /// Deactivates the session once nobody has used the engine for
    /// `deactivationDelay`, with `.notifyOthersOnDeactivation` so audio
    /// the app interrupted (music, a podcast) is told it may resume.
    private func scheduleDeactivation() {
        pendingDeactivation?.cancel()
        pendingDeactivation = nil
        guard sessionActive else { return }
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.pendingDeactivation = nil
            guard self.clients.isEmpty, self.sessionActive else { return }
            do {
                try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            } catch {
                print("[audio] could not deactivate the audio session: \(error.localizedDescription)")
                return
            }
            self.sessionActive = false
            // The next client builds a fresh engine against the session
            // as it is when reactivated: the route may have changed while
            // nothing was listening for it.
            self.built = nil
            print("[audio] audio session deactivated")
        }
        pendingDeactivation = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.deactivationDelay, execute: work)
    }

    private func startIfNeeded() throws {
        if !sessionActive {
            try shapeSession(for: built ?? .none)
        }
        if !engine.isRunning {
            engine.prepare()
            try engine.start()
        }
        routeAwayFromReceiver()
        updateLatency()
        speech.playIfPossible()
        cue.playIfPossible()
    }

    /// Replaces the engine with a fresh one built for `configuration`.
    ///
    /// Order matters and is the documented one: the session is shaped and
    /// active before the new engine exists (so the input node reports the
    /// real hardware format), voice processing is enabled before anything
    /// is connected, and the output graph is connected afterwards.
    private func rebuild(_ configuration: InputConfiguration) throws {
        try shapeSession(for: configuration)

        speech.detachFromEngine()
        cue.detachFromEngine()
        removeTap()
        if !engineIsOrphaned {
            engine.stop()
        }
        if let configurationObserver {
            NotificationCenter.default.removeObserver(configurationObserver)
        }

        let engine = AVAudioEngine()
        self.engine = engine
        engineIsOrphaned = false
        echoCancellationFailed = false
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: .main
        ) { [weak self, weak engine] _ in
            guard let self, let engine, engine === self.engine else { return }
            self.engineConfigurationChanged()
        }

        if configuration != .none {
            let input = engine.inputNode
            if configuration == .voiceProcessed {
                do {
                    try input.setVoiceProcessingEnabled(true)
                    // Other audio (anything not played through this
                    // engine) is ducked as little as the system allows.
                    input.voiceProcessingOtherAudioDuckingConfiguration =
                        AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                            enableAdvancedDucking: false,
                            duckingLevel: .min
                        )
                } catch {
                    // Voice mode still listens, but without echo
                    // cancellation the reply can trip the barge-in
                    // detector; the self-check's no_false_bargein says so.
                    echoCancellationFailed = true
                    print("[audio] voice processing unavailable: \(error.localizedDescription)")
                }
            }
        }

        let submix = AVAudioMixerNode()
        let limiter = AVAudioUnitEffect(audioComponentDescription: Self.peakLimiter)
        engine.attach(submix)
        engine.attach(limiter)
        speech.attach(to: engine, feeding: submix)
        cue.attach(to: engine, feeding: submix)
        engine.connect(submix, to: limiter, format: Self.playbackFormat)
        engine.connect(limiter, to: engine.mainMixerNode, format: Self.playbackFormat)

        let makeUpGain = configuration == .voiceProcessed && !echoCancellationFailed
            ? Self.voiceProcessingMakeUpGain
            : 0
        AudioUnitSetParameter(
            limiter.audioUnit,
            AudioUnitParameterID(kLimiterParam_PreGain),
            kAudioUnitScope_Global,
            0,
            makeUpGain,
            0
        )

        built = configuration
        builtOutputFormat = engine.outputNode.outputFormat(forBus: 0)
        print("[audio] engine built for \(configuration.rawValue) input; "
            + "echo cancellation \(configuration == .voiceProcessed && !echoCancellationFailed ? "on" : "off"), "
            + "make-up gain \(makeUpGain) dB, \(Self.describeRoute())")
    }

    private func installTap() throws -> AVAudioFormat {
        guard let tapBlock else { throw VoiceInputError.noAudioInput }
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        // A zero-channel or zero-rate format means the session handed us
        // no input route (the Simulator without a microphone, a Bluetooth
        // handoff in progress). Installing a tap on it raises; throwing
        // does not.
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw VoiceInputError.noAudioInput
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format, block: tapBlock)
        tapInstalled = true
        tapFormat = format
        return format
    }

    private func removeTap() {
        guard tapInstalled else { return }
        tapInstalled = false
        tapFormat = nil
        if !engineIsOrphaned {
            engine.inputNode.removeTap(onBus: 0)
        }
    }

    // MARK: - Session

    private func shapeSession(for configuration: InputConfiguration) throws {
        let session = AVAudioSession.sharedInstance()
        let mode: AVAudioSession.Mode = configuration == .voiceProcessed ? .voiceChat : .spokenAudio
        var options: AVAudioSession.CategoryOptions = [.defaultToSpeaker, .allowBluetoothA2DP]
        if configuration != .none {
            options.insert(.allowBluetoothHFP)
        }
        if session.category != .playAndRecord || session.mode != mode || session.categoryOptions != options {
            try session.setCategory(.playAndRecord, mode: mode, options: options)
        }
        if !sessionActive {
            try session.setActive(true)
            sessionActive = true
        }
    }

    /// Play-and-record sends output to the earpiece unless told otherwise;
    /// `.defaultToSpeaker` normally handles it, but a reply played into
    /// the earpiece sounds like silence at arm's length, so the route is
    /// checked after every change. Headphones, AirPods and car audio are
    /// never overridden: only the built-in receiver is.
    private func routeAwayFromReceiver() {
        let session = AVAudioSession.sharedInstance()
        guard session.currentRoute.outputs.contains(where: { $0.portType == .builtInReceiver }) else {
            return
        }
        do {
            try session.overrideOutputAudioPort(.speaker)
        } catch {
            print("[audio] could not move output to the speaker: \(error.localizedDescription)")
        }
    }

    private func updateLatency() {
        let session = AVAudioSession.sharedInstance()
        let seconds = session.outputLatency + session.ioBufferDuration
        let frames = AVAudioFramePosition(seconds * Self.playbackFormat.sampleRate)
        speech.latencyFrames = frames
        cue.latencyFrames = frames
    }

    private static func describeRoute() -> String {
        let route = AVAudioSession.sharedInstance().currentRoute
        let inputs = route.inputs.map(\.portType.rawValue).joined(separator: "+")
        let outputs = route.outputs.map(\.portType.rawValue).joined(separator: "+")
        return "route in=\(inputs.isEmpty ? "none" : inputs) out=\(outputs.isEmpty ? "none" : outputs)"
    }

    // MARK: - Interruptions and hardware changes

    @objc private func sessionInterrupted(_ note: Notification) {
        guard
            let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
            AVAudioSession.InterruptionType(rawValue: raw) == .began
        else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            print("[audio] session interrupted")
            // The system has deactivated the session and stopped the
            // engine. Nothing resumes on its own when the interruption
            // ends: the next turn or listening session reactivates.
            self.sessionActive = false
            self.interruptEveryone()
        }
    }

    @objc private func mediaServicesWereReset(_ note: Notification) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            print("[audio] media services were reset")
            self.sessionActive = false
            self.engineIsOrphaned = true
            self.interruptEveryone()
            self.built = nil
        }
    }

    @objc private func routeChanged(_ note: Notification) {
        DispatchQueue.main.async { [weak self] in
            guard let self, self.sessionActive, self.engine.isRunning else { return }
            self.routeAwayFromReceiver()
            self.updateLatency()
        }
    }

    /// The engine stopped itself because the hardware's format changed
    /// (headphones, AirPods, a sample-rate switch). The input tap's format
    /// is stale and the players' queues are not to be trusted, so the
    /// engine is rebuilt for the same configuration with the queued audio
    /// carried over. A notification that changed no format is answered
    /// with the engine it was posted for.
    private func engineConfigurationChanged() {
        print("[audio] engine configuration changed; \(Self.describeRoute())")
        guard let configuration = built else { return }
        // The notification also arrives while the engine keeps running on
        // the formats it was built with: starting a voice-processed engine
        // posts it, and in the Simulator every fresh voice-processed engine
        // does, so rebuilding in answer only posts it again until the loop
        // guard below gives up and voice mode stops. Nothing the graph
        // depends on has changed, so the engine carries on.
        if engine.isRunning, formatsAreUnchanged {
            print("[audio] engine still running on the same formats; keeping it")
            return
        }
        guard !clients.isEmpty else {
            // Nothing is using it: the next user builds against the new
            // hardware.
            built = nil
            return
        }
        // Starting a voice-processed engine can itself switch the hardware
        // rate once, which is one more rebuild and then settles. A route
        // that keeps flapping would rebuild forever; give up instead.
        let now = Date()
        recentRebuilds = recentRebuilds.filter { now.timeIntervalSince($0) < 3 } + [now]
        guard recentRebuilds.count <= 3 else {
            print("[audio] the audio hardware keeps changing; stopping")
            recentRebuilds.removeAll()
            interruptEveryone()
            return
        }
        // Stopped, but on the formats it was built with and with no audio
        // queued: the graph is still valid, so the engine is started again
        // as it is. A fresh engine would only stop itself the same way,
        // which is what a voice-processed engine does in the Simulator on
        // every start.
        if formatsAreUnchanged, !speech.isActive, !cue.isActive {
            do {
                try startIfNeeded()
                print("[audio] engine restarted on the same formats")
                return
            } catch {
                print("[audio] could not restart the engine as it was: \(error.localizedDescription)")
            }
        }
        do {
            try rebuild(configuration)
            if clients.contains(.input) {
                _ = try installTap()
            }
            try startIfNeeded()
        } catch VoiceInputError.noAudioInput where clients.contains(.input) {
            print("[audio] no input route after the configuration change; waiting for the microphone")
            // Whatever is playing carries on meanwhile if the engine will
            // run without its input; if not, it resumes with the retry.
            try? startIfNeeded()
            retryInput(attempt: 1)
        } catch {
            print("[audio] could not restart after a configuration change: \(error.localizedDescription)")
            interruptEveryone()
        }
    }

    /// Tries the microphone again after a configuration change left no
    /// input route. A later configuration change that brings the input
    /// back gets there first through `engineConfigurationChanged`, which
    /// rebuilds with the tap; this covers a route that comes back without
    /// one.
    private func retryInput(attempt: Int) {
        pendingInputRetry?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.pendingInputRetry = nil
            guard self.clients.contains(.input), !self.tapInstalled, let configuration = self.built else { return }
            do {
                try self.rebuild(configuration)
                _ = try self.installTap()
                try self.startIfNeeded()
                print("[audio] microphone input is back")
            } catch VoiceInputError.noAudioInput where attempt < Self.inputRetryLimit {
                try? self.startIfNeeded()
                self.retryInput(attempt: attempt + 1)
            } catch {
                print("[audio] the microphone did not come back: \(error.localizedDescription)")
                self.loseInput()
            }
        }
        pendingInputRetry = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.inputRetryDelay, execute: work)
    }

    /// Gives up the microphone, keeping playback: its user hears about it
    /// through `inputWasLost`, and whatever is playing goes on, on an
    /// engine without the input it no longer has.
    private func loseInput() {
        pendingInputRetry?.cancel()
        pendingInputRetry = nil
        removeTap()
        tapBlock = nil
        clients.remove(.input)
        NotificationCenter.default.post(name: Self.inputWasLost, object: self)
        guard !clients.isEmpty else {
            stopIfIdle()
            return
        }
        do {
            try rebuild(.none)
            try startIfNeeded()
        } catch {
            print("[audio] could not keep playing without the microphone: \(error.localizedDescription)")
            interruptEveryone()
        }
    }

    /// The hardware output and, when the microphone is open, the input
    /// tap still match what the engine was built and tapped with. A
    /// microphone that is claimed but has no tap (waiting for its input
    /// route to come back) is a change still to be dealt with.
    private var formatsAreUnchanged: Bool {
        guard let builtOutputFormat, Self.sameShape(engine.outputNode.outputFormat(forBus: 0), builtOutputFormat) else {
            return false
        }
        guard clients.contains(.input) else { return true }
        guard tapInstalled, let tapFormat else { return false }
        return Self.sameShape(engine.inputNode.outputFormat(forBus: 0), tapFormat)
    }

    private static func sameShape(_ lhs: AVAudioFormat, _ rhs: AVAudioFormat) -> Bool {
        lhs.sampleRate == rhs.sampleRate && lhs.channelCount == rhs.channelCount
    }

    private func interruptEveryone() {
        // Clients stop their channels and release the engine while this
        // posts; whatever is still claimed afterwards is released here.
        NotificationCenter.default.post(name: Self.audioWasInterrupted, object: self)
        pendingInputRetry?.cancel()
        pendingInputRetry = nil
        speech.stop()
        cue.stop()
        removeTap()
        tapBlock = nil
        clients.removeAll()
        stopIfIdle()
    }
}

/// One stream of audio into the shared engine (the spoken reply, or the
/// sample question), with the bookkeeping the engine itself does not keep:
/// what is queued and not yet heard, so it survives an engine rebuild, and
/// how loud each part of it is, so the meter follows what is coming out of
/// the speaker rather than what was handed to the player.
final class PlaybackChannel {

    private struct Item {
        let id: Int
        let buffer: AVAudioPCMBuffer
        /// Normalised 0...1 level of each meter window.
        let levels: [Float]
        /// Where the buffer starts on the player's timeline.
        var start: AVAudioFramePosition
        let completion: () -> Void
    }

    /// 1/60 s at the playback rate: finer than the 30 Hz meter, so the
    /// meter never skips over a syllable.
    private static let meterWindowFrames = 800

    private(set) var player = AVAudioPlayerNode()
    /// Output latency in playback frames, so the meter reads what is
    /// leaving the speaker rather than what was just rendered.
    var latencyFrames: AVAudioFramePosition = 0
    /// Called with true when audio is queued on an idle channel and with
    /// false when the channel drains or stops.
    var activityChanged: ((Bool) -> Void)?

    private var items: [Item] = []
    private var nextID = 0
    /// Bumped whenever the player is stopped or replaced, so completions a
    /// stopped player fires for buffers it dropped are ignored.
    private var epoch = 0

    /// Audio is queued that has not finished playing.
    var isActive: Bool { !items.isEmpty }

    /// Volume of this channel in the mix, 0...1. Kept here rather than
    /// only on the player because an engine rebuild replaces the player.
    var gain: Float = 1 {
        didSet { player.volume = gain }
    }

    /// Queues `buffer` (in `AudioEngineHub.playbackFormat`) after anything
    /// already queued and starts playing if the engine is running.
    /// `completion` runs on the main queue once the buffer has been heard;
    /// it never runs for audio dropped by `stop()`.
    func schedule(_ buffer: AVAudioPCMBuffer, completion: @escaping () -> Void) {
        let wasActive = isActive
        let start: AVAudioFramePosition
        if let last = items.last {
            start = last.start + AVAudioFramePosition(last.buffer.frameLength)
        } else {
            // Nothing queued: a buffer scheduled now starts at the
            // player's current position.
            start = renderPosition() ?? 0
        }
        let levels = AudioLevel.windowRMS(buffer, windowFrames: Self.meterWindowFrames).map {
            AudioLevel.normalized(decibels: AudioLevel.decibels(rms: $0))
        }
        nextID += 1
        let item = Item(id: nextID, buffer: buffer, levels: levels, start: start, completion: completion)
        items.append(item)
        enqueueOnPlayer(item)
        if !wasActive {
            activityChanged?(true)
        }
        playIfPossible()
    }

    /// Silences the channel at once and forgets everything queued.
    func stop() {
        epoch += 1
        let wasActive = isActive
        items.removeAll()
        player.stop()
        if wasActive {
            activityChanged?(false)
        }
    }

    /// Loudness of what is coming out of the speaker on this channel now.
    var level: Float {
        guard let rendered = renderPosition() else { return 0 }
        let position = rendered - latencyFrames
        for item in items where position >= item.start {
            let window = Int(position - item.start) / Self.meterWindowFrames
            if window < item.levels.count {
                return item.levels[window]
            }
        }
        return 0
    }

    /// Starts the player when it has something to play and its engine is
    /// running. `play()` on a node whose engine is stopped raises, so this
    /// is the only place that calls it.
    func playIfPossible() {
        guard !items.isEmpty, !player.isPlaying, player.engine?.isRunning == true else { return }
        player.play()
    }

    // MARK: - Engine rebuilds

    fileprivate func detachFromEngine() {
        epoch += 1
        player.stop()
    }

    /// Puts a fresh player on `engine` and queues on it everything that
    /// had not been heard yet. A buffer the old player was halfway through
    /// plays again from its start, which is why queued audio is kept in
    /// short pieces.
    fileprivate func attach(to engine: AVAudioEngine, feeding mixer: AVAudioMixerNode) {
        player = AVAudioPlayerNode()
        player.volume = gain
        engine.attach(player)
        engine.connect(
            player,
            to: mixer,
            fromBus: 0,
            toBus: mixer.nextAvailableInputBus,
            format: AudioEngineHub.playbackFormat
        )
        var position: AVAudioFramePosition = 0
        for index in items.indices {
            items[index].start = position
            position += AVAudioFramePosition(items[index].buffer.frameLength)
            enqueueOnPlayer(items[index])
        }
    }

    // MARK: - Internals

    private func enqueueOnPlayer(_ item: Item) {
        let epoch = self.epoch
        let id = item.id
        player.scheduleBuffer(
            item.buffer,
            at: nil,
            options: [],
            completionCallbackType: .dataPlayedBack
        ) { [weak self] _ in
            DispatchQueue.main.async {
                self?.didPlay(id, epoch: epoch)
            }
        }
    }

    private func didPlay(_ id: Int, epoch: Int) {
        guard epoch == self.epoch, let index = items.firstIndex(where: { $0.id == id }) else { return }
        // Buffers play in the order they were queued, so anything ahead of
        // this one has been heard too, even if its callback was lost.
        let finished = items[...index]
        items.removeFirst(index + 1)
        for item in finished {
            item.completion()
        }
        if items.isEmpty {
            activityChanged?(false)
        }
    }

    private func renderPosition() -> AVAudioFramePosition? {
        guard
            player.isPlaying,
            let nodeTime = player.lastRenderTime,
            nodeTime.isSampleTimeValid,
            let playerTime = player.playerTime(forNodeTime: nodeTime)
        else { return nil }
        return playerTime.sampleTime
    }
}

/// Whether a reply is coming out of the speaker, or was a moment ago:
/// room reverberation keeps the residual echo up for a few hundred
/// milliseconds after the last sample plays. Read on the input tap's
/// thread, written on the main thread.
final class EchoGate {
    private static let tail: CFAbsoluteTime = 0.35

    private let lock = NSLock()
    private var sources: Set<ObjectIdentifier> = []
    private var quietSince: CFAbsoluteTime = 0

    func set(_ source: ObjectIdentifier, active: Bool) {
        lock.lock()
        defer { lock.unlock() }
        let wasActive = !sources.isEmpty
        if active {
            sources.insert(source)
        } else {
            sources.remove(source)
        }
        if wasActive, sources.isEmpty {
            quietSince = CFAbsoluteTimeGetCurrent()
        }
    }

    var isEchoLikely: Bool {
        lock.lock()
        defer { lock.unlock() }
        return !sources.isEmpty || CFAbsoluteTimeGetCurrent() - quietSince < Self.tail
    }
}
