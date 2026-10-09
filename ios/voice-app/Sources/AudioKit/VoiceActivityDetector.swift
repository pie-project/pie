import Foundation

/// Decides from the echo-cancelled microphone level alone when the user
/// starts and stops talking.
///
/// Voice mode gates recognition on it: nothing goes to the recogniser
/// until onset, so each recognition task covers one utterance (well inside
/// the one-minute task limit) and a quiet room costs nothing. It is a
/// level detector against measured references, not a speech classifier,
/// so its thresholds are relative to whatever the room sounds like, with
/// absolute minimums so a silent room does not make it hair-triggered.
///
/// Two references are kept, each learnt only from what is not the user:
///   - the room's noise floor, while no reply is playing;
///   - the residue the echo canceller leaves of a reply, while one is
///     playing, carried from one reply to the next. It is louder than the
///     room and follows the reply's syllables, so it is tracked as an
///     envelope over the loud parts of those syllables, and onset during a
///     reply needs a voice clearly above that envelope as well as above
///     the room, and never below an absolute minimum. The user talking
///     over the reply is well above it.
/// Keeping them apart matters: folded into one floor, the residue would
/// raise the room floor and the echo margin would then be stacked on top
/// of it, which put barge-in out of reach of a normal speaking voice.
/// In the moments after a reply stops (`PlaybackEcho.tail`), onset is
/// still held to the reply's thresholds, but neither reference learns:
/// what is heard then is the reply's last reverberation, or the user
/// answering it.
///
/// Neither reference is assumed. A reference that starts from a guess and
/// only learns from levels below the threshold it sets can never learn a
/// room, or a residue, louder than that guess: a steady -45 dBFS room read
/// as a voice that never stopped, and a loud residue as the user barging
/// in on every reply. So:
///   - the floor is measured over the first moments of every listening
///     session, before anything can count as speech;
///   - the residue is measured over the first audible moments of the first
///     reply heard on a route, before anything can count as talking over
///     it. Later replies start from what the earlier ones left, so the
///     user can talk over them from their first syllable, and a user who
///     keeps talking as a reply starts is not taken for its residue;
///   - a level that holds still for most of a second is a noise (a fan, a
///     hum), never a voice, which rises and falls with every syllable.
///     Whichever reference is in force follows it up quickly even above
///     the threshold, so a noise that starts mid-session ends the
///     "utterance" it began within a couple of seconds and starts no more.
///
/// The detector outlives a listening session: the residue it has learnt is
/// a property of the route and the volume, not of the session, and is
/// forgotten only when the route changes (`forgetRoute()`).
///
/// The numbers are for the iPhone's voice-processed microphone (automatic
/// gain control on), set from what the audio self-check measured there and
/// checked by its no_false_bargein, echo_transcripts and external_onset
/// results. The session's echo-cancelled input (`-PieEchoPath session`)
/// has no gain control and has not been measured; its self-check results
/// say whether they hold there too.
struct VoiceActivityDetector {

    enum Event: Equatable {
        case onset
        case end
    }

    /// Onset needs the level this far above the room floor...
    static let onsetMargin: Float = 10
    /// ...and at least this loud, in dBFS.
    static let onsetMinimum: Float = -50
    /// How long the level has to stay up: long enough to ignore a click
    /// or a cough's first burst, short enough not to lose the first word
    /// (the pre-roll covers it).
    static let onsetTime: TimeInterval = 0.2

    /// While a reply plays or has just stopped, onset needs the level above
    /// all three of the thresholds below, for `onsetTime` as without one.
    ///
    /// Onset over a reply decides nothing by itself: voice mode ducks the
    /// reply and lets `EchoFilter` judge the words recognised. An onset the
    /// residue sets off costs a second or two of quieter reply; a voice
    /// missed costs the user their interruption. The level alone cannot
    /// tell them apart: what leaks through the echo canceller comes a
    /// syllable or a word at a time, sometimes louder than the user, and
    /// "stop" said over the reply is one syllable too. Asking for 0.35 s
    /// rather than 0.2 kept out most of the leaks, and nearly every
    /// one-word interruption with them, so only the thresholds are stricter
    /// than without a reply.
    ///
    /// The audio self-check on an iPhone 16 Pro in a quiet room, with the
    /// reply at the earlier +6 dB make-up gain, measured the residue
    /// reaching the detector: a median near -38 dBFS, a 90th percentile
    /// between -34.6 and -31.4 dBFS, bursts as loud as -8 dBFS, and an
    /// onset during the reply in three of four runs; a normal voice at arm's
    /// length measures about -30 to -15 dBFS. Run against level tracks
    /// shaped like that residue (words of syllables and gaps, some leaking
    /// louder), these thresholds against the earlier ones (a 6 dB residue
    /// margin and a -38 dBFS minimum), on the same residue:
    ///   - a 7 s reply whose leaks reach 16 dB over its usual syllables set
    ///     off onset in 10 to 14% of replies, against 20 to 29%; with leaks
    ///     as loud as the -8 dBFS measured, in about half either way;
    ///   - a voice talking over the reply at -22 dBFS was missed within 4 s
    ///     in 17% of tries against 5% with the residue 4 dB lower, as the
    ///     smaller make-up gain should leave it, and in 59% against 35%
    ///     with the residue as measured;
    ///   - "stop" at -18 dBFS, its vowel 233 ms long, was heard in 95 to
    ///     99% of tries, as before.
    /// So the phone has to show the residue coming down with the make-up
    /// gain; if it does not, the residue margin is the number to give
    /// back first.
    ///
    /// This far above the room floor.
    static let echoOnsetMargin: Float = 18
    /// This far above the residue envelope, which settles near the
    /// residue's 95th percentile (it jumps up to loud windows and lets go
    /// over ten seconds), some 6 dB over its median.
    static let echoResidueMargin: Float = 7
    /// At least this loud, in dBFS: about the residue's 90th percentile as
    /// measured, so most of it is kept out while the envelope is still
    /// being learnt or is out of date, and under the quiet end of a normal
    /// voice.
    static let echoOnsetMinimum: Float = -32

    /// Which echo canceller cleaned the input; set for every buffer from
    /// what the engine is built on. The echo thresholds above were tuned
    /// and measured on voice processing. The session's echo-cancelled
    /// input leaves far less of a reply behind (on an iPhone 16 Pro the
    /// loudest residue window measured -40 dBFS against -13 dBFS with
    /// voice processing, its 90th percentile -48 against -41) and has no
    /// automatic gain control, so a voice also arrives several dB quieter.
    /// There, the parts of the threshold that stand in for an unknown
    /// residue come down; the margin over the learnt residue envelope is
    /// the same.
    enum Canceller: Equatable {
        case voiceProcessing
        case session
    }

    var canceller: Canceller = .voiceProcessing

    /// The session canceller's room margin and absolute minimum.
    static let sessionEchoOnsetMargin: Float = 12
    static let sessionEchoOnsetMinimum: Float = -40
    /// Once talking, the level only has to stay this far above the room
    /// floor to count as still talking (and, while a reply plays or has
    /// just stopped, above the loud parts of its residue). Lower than the onset margin, so the
    /// quiet end of a word does not count as silence.
    static let holdMargin: Float = 6
    static let holdMinimum: Float = -56
    /// Silence that ends an utterance.
    static let trailingSilence: TimeInterval = 1.1
    /// Ends an utterance that has not ended by itself, well inside the
    /// recogniser's one-minute task limit.
    static let maximumUtterance: TimeInterval = 45
    /// The room is measured for this long, from levels with no reply
    /// playing, at the start of every session. The quietest window is the
    /// floor, so a user who starts talking at once does not become the
    /// room; a first word spoken meanwhile is still in the pre-roll.
    static let floorSeedTime: TimeInterval = 0.3
    /// When the first reply on a route starts, its residue is measured
    /// until this much of it has been heard (windows `audibleMargin` above
    /// the room), or for `residueSeedLimit` if the echo canceller leaves
    /// nothing to hear. The loudest window is the residue. Counting only
    /// audible windows gets past the synthesizer's leading silence, which
    /// would otherwise be measured as the residue; nobody talks over the
    /// first second of the first answer they hear.
    static let residueSeedTime: TimeInterval = 0.4
    static let residueSeedLimit: TimeInterval = 1.2
    static let audibleMargin: Float = 6
    /// A level that stays within this range for `steadyTime` is a steady
    /// noise. Speech moves through far more than this between syllables
    /// in under a second.
    static let steadyRange: Float = 6
    static let steadyTime: TimeInterval = 0.8
    /// How quickly a reference follows a steady noise above it.
    private static let steadyFollowTime: TimeInterval = 0.5

    private static let initialFloor: Float = -60
    private static let floorRange: ClosedRange<Float> = -90 ... -20

    /// The room's level when nobody is talking and nothing is playing,
    /// in dBFS.
    private(set) var floor = VoiceActivityDetector.initialFloor
    /// The level of a playing reply's echo residue, in dBFS.
    private(set) var echoResidue = VoiceActivityDetector.initialFloor
    private(set) var isSpeech = false

    /// The level onset needs while a reply plays, from what has been
    /// learnt so far, in dBFS. Read by the audio self-check.
    var echoOnsetLevel: Float { onsetThreshold(echoLikely: true) }

    private var floorSeeded = false
    private var floorMeasured: TimeInterval = 0
    private var floorSeedLevel = Float.infinity
    /// A reply's residue has been measured on this route.
    private var residueKnown = false
    /// Measuring the residue of the reply that just started.
    private var residueSeeding = false
    private var residueHeard: TimeInterval = 0
    private var residueWaited: TimeInterval = 0
    private var residueSeedLevel = -Float.infinity
    /// The last `steadyTime` of levels, oldest first.
    private var recent: [(level: Float, duration: TimeInterval)] = []
    private var recentDuration: TimeInterval = 0
    private var onsetEvidence: TimeInterval = 0
    private var silence: TimeInterval = 0
    private var speechTime: TimeInterval = 0
    private var wasEchoLikely = false

    /// Feeds one window's level (dBFS) lasting `duration` seconds, with
    /// what the speaker was doing when it was heard.
    mutating func process(level: Float, duration: TimeInterval, echo: PlaybackEcho) -> Event? {
        let steady = remember(level, duration: duration)
        let echoLikely = echo != .none
        if echoLikely, !wasEchoLikely {
            // A reply just started; its residue is never quieter than the
            // room. If it is the first on this route, it is measured before
            // anything counts as talking over it, unless the user is
            // already talking.
            echoResidue = max(echoResidue, floor)
            residueSeeding = !residueKnown && !isSpeech
            residueHeard = 0
            residueWaited = 0
            residueSeedLevel = -.infinity
        } else if !echoLikely, residueSeeding {
            // The reply was over before the measurement was: what was
            // heard of it still counts, and the next reply measures again.
            echoResidue = Self.clamped(max(echoResidue, residueSeedLevel))
            residueSeeding = false
        }
        wasEchoLikely = echoLikely

        if isSpeech {
            return continueSpeech(level: level, duration: duration, echo: echo, steady: steady)
        }

        if echoLikely, residueSeeding {
            // Only what is heard while the reply plays is measured: in the
            // moments after it the user may already be answering.
            if echo == .playing {
                residueWaited += duration
                if level >= floor + Self.audibleMargin {
                    residueHeard += duration
                    residueSeedLevel = max(residueSeedLevel, level)
                }
                if residueHeard >= Self.residueSeedTime || residueWaited >= Self.residueSeedLimit {
                    echoResidue = Self.clamped(max(echoResidue, residueSeedLevel))
                    residueSeeding = false
                    residueKnown = true
                }
            }
            onsetEvidence = 0
            return nil
        }
        if !echoLikely, !floorSeeded {
            floorSeedLevel = min(floorSeedLevel, level)
            floorMeasured += duration
            if floorMeasured >= Self.floorSeedTime {
                floor = Self.clamped(floorSeedLevel)
                floorSeeded = true
            }
            onsetEvidence = 0
            return nil
        }

        if level > onsetThreshold(echoLikely: echoLikely) {
            if steady {
                // Loud but unchanging: a noise that has become part of the
                // room, or of the residue, not someone starting to talk.
                onsetEvidence = max(0, onsetEvidence - duration)
                switch echo {
                case .none:
                    adapt(&floor, toward: level, duration: duration, timeConstant: Self.steadyFollowTime)
                case .playing:
                    adapt(&echoResidue, toward: level, duration: duration, timeConstant: Self.steadyFollowTime)
                case .tail:
                    break
                }
                return nil
            }
            onsetEvidence += duration
            if onsetEvidence >= Self.onsetTime {
                isSpeech = true
                speechTime = onsetEvidence
                onsetEvidence = 0
                silence = 0
                return .onset
            }
            return nil
        }

        // Leaky rather than reset, so the dip between two syllables does
        // not throw away the first one.
        onsetEvidence = max(0, onsetEvidence - duration)
        switch echo {
        case .none:
            // Falls quickly (a noise ending) and rises slowly.
            adapt(&floor, toward: level, duration: duration, timeConstant: level < floor ? 0.15 : 2)
        case .playing:
            // Below the threshold, so it is taken for residue: the envelope
            // jumps to the residue's loud parts and lets go of them slowly,
            // so the gaps between the reply's words do not pull it down to
            // where its next syllable would read as the user. A syllable
            // louder than the threshold is rejected by the onset evidence
            // leaking away before it is over.
            adapt(&echoResidue, toward: level, duration: duration, timeConstant: level > echoResidue ? 0.15 : 10)
        case .tail:
            // The reply has stopped, and what is heard now is its last
            // reverberation or the user answering it, often quietly at
            // first. Learnt as residue, an answer like that would raise
            // the envelope that the next reply's barge-in has to clear;
            // learnt as the room, the reverberation would raise the floor.
            // So neither reference moves.
            break
        }
        return nil
    }

    /// Back to waiting for onset, keeping what was learnt about the room.
    mutating func endSpeech() {
        isSpeech = false
        onsetEvidence = 0
        silence = 0
        speechTime = 0
    }

    /// A new listening session on the same route: the room is measured
    /// again (it may be a different room by now); the residue is kept.
    mutating func restart() {
        endSpeech()
        floorSeeded = false
        floorMeasured = 0
        floorSeedLevel = .infinity
        residueSeeding = false
        residueHeard = 0
        residueWaited = 0
        residueSeedLevel = -.infinity
        recent.removeAll()
        recentDuration = 0
        wasEchoLikely = false
    }

    /// The audio route changed (a headset connected or went away): the
    /// residue belonged to the old one, and the room sounds different
    /// through the new microphone.
    mutating func forgetRoute() {
        restart()
        echoResidue = Self.initialFloor
        residueKnown = false
    }

    // MARK: - Internals

    private mutating func continueSpeech(level: Float, duration: TimeInterval, echo: PlaybackEcho, steady: Bool) -> Event? {
        speechTime += duration
        var hold = max(floor + Self.holdMargin, Self.holdMinimum)
        if echo != .none {
            // A reply's residue is louder than the room, so above the room
            // is not enough to be still talking: above the residue's loud
            // parts is. Otherwise an onset the residue itself set off is
            // held open by it for the rest of the reply and past its end,
            // feeding the recogniser the reply's words all the way, and an
            // utterance that outlives the reply reaches voice mode as a
            // new question with no reply left to compare it with. Voice
            // mode ducks the reply on onset, which drops the residue well
            // under this, so such an utterance ends one trailing silence
            // later, while the reply is still there to compare it with.
            // A voice talking over the reply rises well above it on every
            // syllable, so for the user only a real pause counts.
            hold = max(hold, echoResidue)
        }
        if level > hold {
            silence = 0
        } else {
            silence += duration
        }
        switch echo {
        case .none:
            // A noise that has become part of the room (a fan switched on)
            // must not hold an utterance open: the floor rises underneath
            // it, within about a second once the level is plainly steady
            // and only very slowly otherwise, so it never follows a voice,
            // and falls back quickly when the noise stops.
            let timeConstant = level < floor ? 0.5 : (steady ? Self.steadyFollowTime : 15)
            adapt(&floor, toward: level, duration: duration, timeConstant: timeConstant)
        case .playing:
            if steady, level > echoResidue {
                adapt(&echoResidue, toward: level, duration: duration, timeConstant: Self.steadyFollowTime)
            }
        case .tail:
            break
        }
        if silence >= Self.trailingSilence || speechTime >= Self.maximumUtterance {
            endSpeech()
            return .end
        }
        return nil
    }

    private func onsetThreshold(echoLikely: Bool) -> Float {
        guard echoLikely else {
            return max(floor + Self.onsetMargin, Self.onsetMinimum)
        }
        switch canceller {
        case .voiceProcessing:
            return max(floor + Self.echoOnsetMargin, echoResidue + Self.echoResidueMargin, Self.echoOnsetMinimum)
        case .session:
            return max(
                floor + Self.sessionEchoOnsetMargin,
                echoResidue + Self.echoResidueMargin,
                Self.sessionEchoOnsetMinimum
            )
        }
    }

    /// Adds a window to the recent history and says whether the history
    /// is a full `steadyTime` long and has stayed within `steadyRange`.
    private mutating func remember(_ level: Float, duration: TimeInterval) -> Bool {
        recent.append((level, duration))
        recentDuration += duration
        while let first = recent.first, recentDuration - first.duration >= Self.steadyTime {
            recent.removeFirst()
            recentDuration -= first.duration
        }
        // A whole window's worth, allowing for the rounding of window
        // lengths that do not divide it exactly.
        guard recentDuration >= Self.steadyTime * 0.95 else { return false }
        var lowest = Float.infinity
        var highest = -Float.infinity
        for entry in recent {
            lowest = min(lowest, entry.level)
            highest = max(highest, entry.level)
        }
        return highest - lowest < Self.steadyRange
    }

    private func adapt(_ value: inout Float, toward level: Float, duration: TimeInterval, timeConstant: TimeInterval) {
        let alpha = Float(1 - exp(-duration / timeConstant))
        value = Self.clamped(value + (level - value) * alpha)
    }

    private static func clamped(_ value: Float) -> Float {
        min(max(value, floorRange.lowerBound), floorRange.upperBound)
    }
}
