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
/// The numbers are starting points for the iPhone's voice-processed
/// microphone (automatic gain control on), to be checked against the audio
/// self-check's no_false_bargein and external_onset results.
struct VoiceActivityDetector {

    enum Event: Equatable {
        case onset
        case end
    }

    /// Onset needs the level this far above the room floor...
    static let onsetMargin: Float = 10
    /// ...and at least this loud, in dBFS.
    static let onsetMinimum: Float = -50
    /// While a reply plays: this far above the room floor, this far above
    /// the loud parts of the echo residue, and at least this loud.
    static let echoOnsetMargin: Float = 18
    static let echoResidueMargin: Float = 6
    static let echoOnsetMinimum: Float = -38
    /// How long the level has to stay up: long enough to ignore a click
    /// or a cough's first burst, short enough not to lose the first word
    /// (the pre-roll covers it).
    static let onsetTime: TimeInterval = 0.2
    /// Once talking, the level only has to stay this far above the room
    /// floor to count as still talking. Lower than the onset margin, so
    /// the quiet end of a word does not count as silence.
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

    /// Feeds one window's level (dBFS) lasting `duration` seconds.
    mutating func process(level: Float, duration: TimeInterval, echoLikely: Bool) -> Event? {
        let steady = remember(level, duration: duration)
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
            return continueSpeech(level: level, duration: duration, echoLikely: echoLikely, steady: steady)
        }

        if echoLikely, residueSeeding {
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
                if echoLikely {
                    adapt(&echoResidue, toward: level, duration: duration, timeConstant: Self.steadyFollowTime)
                } else {
                    adapt(&floor, toward: level, duration: duration, timeConstant: Self.steadyFollowTime)
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
        if echoLikely {
            // Below the threshold, so it cannot be the user: the envelope
            // jumps to the residue's loud parts and lets go of them slowly,
            // so the gaps between the reply's words do not pull it down to
            // where its next syllable would read as the user. A syllable
            // louder than the threshold is rejected by the onset evidence
            // leaking away before it is over.
            adapt(&echoResidue, toward: level, duration: duration, timeConstant: level > echoResidue ? 0.15 : 10)
        } else {
            // Falls quickly (a noise ending) and rises slowly.
            adapt(&floor, toward: level, duration: duration, timeConstant: level < floor ? 0.15 : 2)
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

    private mutating func continueSpeech(level: Float, duration: TimeInterval, echoLikely: Bool, steady: Bool) -> Event? {
        speechTime += duration
        let hold = max(floor + Self.holdMargin, Self.holdMinimum)
        if level > hold {
            silence = 0
        } else {
            silence += duration
        }
        if !echoLikely {
            // A noise that has become part of the room (a fan switched on)
            // must not hold an utterance open: the floor rises underneath
            // it, within about a second once the level is plainly steady
            // and only very slowly otherwise, so it never follows a voice,
            // and falls back quickly when the noise stops.
            let timeConstant = level < floor ? 0.5 : (steady ? Self.steadyFollowTime : 15)
            adapt(&floor, toward: level, duration: duration, timeConstant: timeConstant)
        } else if steady, level > echoResidue {
            adapt(&echoResidue, toward: level, duration: duration, timeConstant: Self.steadyFollowTime)
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
        return max(floor + Self.echoOnsetMargin, echoResidue + Self.echoResidueMargin, Self.echoOnsetMinimum)
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
