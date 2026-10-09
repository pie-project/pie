import SwiftUI

/// The sphere at the centre of voice mode, after ChatGPT's advanced voice
/// orb: a glass ball of soft blue-and-white sky whose clouds drift and
/// slowly swirl, with a slightly brighter rim. It is never quite still.
/// The colors are Pie's (the Yale blues and white), not OpenAI's.
///
/// How it behaves, mood by mood (ChatGPT's app, estimated by eye):
/// - connecting: dimmer and smaller, with a gentle pulse;
/// - listening: breathes (1.0 to 1.03 over 3.5 s) and drifts slowly, and
///   swells a little while the user talks;
/// - thinking: a slower, softer pulse, a touch smaller, the drift slowed
///   to a holding pattern;
/// - speaking: grows with the voice, to about 1.15 at its loudest, and
///   its clouds move two to three times faster;
/// - muted and failed: washed out and deaf to the room, but still idling.
///
/// None of this is SwiftUI animation. `OrbDynamics` works out each frame
/// from the time since the last one: it eases the look towards the
/// mood's, so a change of mood never snaps (even halfway through another);
/// it follows the loudness with a 50 ms attack and a 200 ms release, so it
/// never jitters; and it advances the drift by speed × time, so a change
/// of speed never makes the clouds jump. Loudness is read from the
/// `VoiceLevelMeter` each frame instead of being passed in, so the audio
/// callbacks never redraw anything; only this view's timeline does.
///
/// Reduce Motion: no drift and no change of size. The orb breathes in
/// opacity only and shows loudness as brightness.
struct VoiceOrb: View {

    enum Mood: Equatable {
        /// Voice mode is starting and the microphone is not open yet.
        case connecting
        case listening
        case thinking
        case speaking
        /// The microphone is off.
        case muted
        case failed
    }

    let mood: Mood
    /// Where the orb reads loudness each frame; nil for an orb that does
    /// not react to sound.
    var levels: VoiceLevelMeter?
    var diameter: CGFloat = 240

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.scenePhase) private var scenePhase
    @State private var dynamics = OrbDynamics()
    /// The timeline only runs while the orb is on screen and the app is in
    /// front, so it costs nothing once voice mode has closed.
    @State private var isOnScreen = false

    var body: some View {
        TimelineView(.animation(minimumInterval: reduceMotion ? 1.0 / 30 : nil, paused: !isOnScreen || scenePhase != .active)) { timeline in
            let frame = dynamics.advance(
                to: timeline.date.timeIntervalSinceReferenceDate,
                mood: mood,
                levels: levels,
                reduceMotion: reduceMotion
            )
            OrbSphere(frame: frame, diameter: diameter)
        }
        .frame(width: diameter, height: diameter)
        .onAppear { isOnScreen = true }
        .onDisappear { isOnScreen = false }
    }
}

/// One frame of the orb: everything `OrbSphere` needs to draw it.
struct OrbFrame {
    /// Size relative to the diameter, breathing and loudness included.
    var scale: Double = 1
    /// How far the clouds have drifted, in radians.
    var drift: Double = 0
    var opacity: Double = 1
    var saturation: Double = 1
    var brightness: Double = 0
}

// MARK: - Drawing

/// The orb drawn at one moment. On iOS 18 the sky is a `MeshGradient`
/// whose inner points wander, so its colors billow; on iOS 17 it is two
/// blurred angular gradients turning against each other over a vertical
/// one. Either way, white clouds drift over it and the rim is lighter.
struct OrbSphere: View {
    let frame: OrbFrame
    let diameter: CGFloat

    var body: some View {
        ZStack {
            if #available(iOS 18.0, *) {
                meshSky
            } else {
                layeredSky
            }
            clouds
            rim
        }
        .clipShape(Circle())
        // Flattened into one layer: cheaper to draw every frame, and the
        // opacity below fades the orb as a whole instead of each layer.
        .drawingGroup()
        .saturation(frame.saturation)
        .brightness(frame.brightness)
        .opacity(frame.opacity)
        .scaleEffect(frame.scale)
    }

    /// The whole sky turns, slower than its clouds drift.
    private var swirl: Angle { .radians(frame.drift * 0.3) }

    /// Halfway between Yale light blue and white: the pale sky between
    /// the clouds.
    private static let sky = Color(hex: 0xB1D4FF)
    /// A quarter of the way from white to Yale light blue: the rim.
    private static let rimLight = Color(hex: 0xD8EAFF)

    // MARK: Sky, iOS 18

    @available(iOS 18.0, *)
    private var meshSky: some View {
        // Drawn half again as large as the orb, so the corners of the
        // square never show inside the circle as it turns.
        MeshGradient(width: 4, height: 4, points: Self.meshPoints(drift: frame.drift), colors: Self.meshColors)
            .scaleEffect(1.5)
            .rotationEffect(swirl)
    }

    /// Row by row, top to bottom. The circle shows mostly the four inner
    /// points and the middles of the edges.
    private static let meshColors: [Color] = [
        .white, sky, .white, sky,
        sky, Theme.yaleLight, .white, Theme.yaleLight,
        Theme.yaleLight, sky, Theme.yaleMedium, sky,
        Theme.yaleMedium, Theme.yaleLight, sky, Theme.yaleMedium,
    ]

    /// A 4 × 4 grid whose four inner points wander on slow loops of their
    /// own. The border stays put, so the mesh always fills its square, and
    /// no point strays more than 0.12 from home, so neighbours never cross.
    private static func meshPoints(drift: Double) -> [SIMD2<Float>] {
        var points: [SIMD2<Float>] = []
        for row in 0..<4 {
            for column in 0..<4 {
                var x = Double(column) / 3
                var y = Double(row) / 3
                let isInner = (1...2).contains(row) && (1...2).contains(column)
                if isInner {
                    let seed = Double(row * 4 + column)
                    x += 0.12 * sin(drift * (0.8 + 0.06 * seed) + seed)
                    y += 0.12 * cos(drift * (0.7 + 0.05 * seed) + seed * 1.7)
                }
                points.append(SIMD2(Float(x), Float(y)))
            }
        }
        return points
    }

    // MARK: Sky, iOS 17

    private var layeredSky: some View {
        ZStack {
            Circle()
                .fill(LinearGradient(colors: [Self.sky, Theme.yaleLight, Theme.yaleMedium], startPoint: .top, endPoint: .bottom))
            Circle()
                .fill(AngularGradient(colors: [Theme.yaleLight, .white, Self.sky, Theme.yaleMedium, Theme.yaleLight], center: .center))
                .rotationEffect(swirl * 1.6)
                .blur(radius: diameter * 0.12)
                .opacity(0.6)
            Circle()
                .fill(AngularGradient(colors: [.clear, .white.opacity(0.9), .clear, Theme.yaleMedium.opacity(0.7), .clear], center: .center))
                .rotationEffect(-swirl * 2.3)
                .blur(radius: diameter * 0.16)
                .opacity(0.5)
        }
    }

    // MARK: Clouds and rim

    /// Three soft white clouds on slow loops of different lengths, so
    /// their pattern never visibly repeats.
    private var clouds: some View {
        let t = frame.drift
        return ZStack {
            cloud(width: 0.62, height: 0.34, x: 0.18 * sin(t * 0.9), y: -0.17 + 0.09 * cos(t * 0.7), opacity: 0.6)
            cloud(width: 0.50, height: 0.30, x: -0.20 * cos(t * 1.1 + 1), y: 0.10 + 0.08 * sin(t * 0.8 + 2), opacity: 0.4)
            cloud(width: 0.40, height: 0.24, x: 0.22 * sin(t * 0.6 + 4), y: 0.20 * cos(t * 0.5 + 1), opacity: 0.35)
        }
        .blur(radius: diameter * 0.075)
    }

    /// One cloud; sizes and offsets are fractions of the diameter.
    private func cloud(width: Double, height: Double, x: Double, y: Double, opacity: Double) -> some View {
        Ellipse()
            .fill(Color.white.opacity(opacity))
            .frame(width: diameter * width, height: diameter * height)
            .offset(x: diameter * x, y: diameter * y)
    }

    /// The lighter rim: a glow inside the edge and a soft line on it. Pale
    /// blue rather than white, so the edge still shows on a white page.
    private var rim: some View {
        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        stops: [
                            .init(color: Self.rimLight.opacity(0), location: 0.72),
                            .init(color: Self.rimLight.opacity(0.5), location: 1),
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: diameter / 2
                    )
                )
            Circle()
                .strokeBorder(Self.rimLight.opacity(0.7), lineWidth: diameter * 0.014)
                .blur(radius: diameter * 0.008)
        }
    }
}

// MARK: - Motion

/// The orb's moving state, advanced once per frame by `advance(to:…)`.
/// A plain class kept in `@State` rather than state itself: it changes up
/// to 120 times a second and only the frame being drawn needs it, so
/// changing it must not ask SwiftUI to redraw anything.
@MainActor
final class OrbDynamics {

    /// What a mood looks like, at rest.
    struct Look {
        /// Size, relative to the diameter.
        var scale: Double
        /// How far each breath swells it: 0.03 is 3%.
        var breath: Double
        /// One breath, in seconds.
        var breathPeriod: Double
        /// How fast the clouds drift, in cycles a second.
        var drift: Double
        var opacity: Double
        var saturation: Double

        /// Estimated from ChatGPT's orb: drift about 0.05-0.1 cycles a
        /// second at rest, two to three times that while speaking.
        static func of(_ mood: VoiceOrb.Mood) -> Look {
            switch mood {
            case .connecting:
                return Look(scale: 0.82, breath: 0.04, breathPeriod: 1.6, drift: 0.05, opacity: 0.7, saturation: 0.9)
            case .listening:
                return Look(scale: 1, breath: 0.03, breathPeriod: 3.5, drift: 0.075, opacity: 1, saturation: 1)
            case .thinking:
                return Look(scale: 0.95, breath: 0.02, breathPeriod: 2.6, drift: 0.045, opacity: 0.92, saturation: 1)
            case .speaking:
                return Look(scale: 1, breath: 0.01, breathPeriod: 3.5, drift: 0.19, opacity: 1, saturation: 1)
            case .muted:
                return Look(scale: 0.97, breath: 0.03, breathPeriod: 3.5, drift: 0.06, opacity: 0.9, saturation: 0.12)
            case .failed:
                return Look(scale: 0.9, breath: 0.015, breathPeriod: 4.5, drift: 0.03, opacity: 0.75, saturation: 0.15)
            }
        }

        /// This look moved the fraction `amount` (0...1) of the way to
        /// `target`.
        func moved(toward target: Look, by amount: Double) -> Look {
            func step(_ from: Double, _ to: Double) -> Double { from + (to - from) * amount }
            return Look(
                scale: step(scale, target.scale),
                breath: step(breath, target.breath),
                breathPeriod: step(breathPeriod, target.breathPeriod),
                drift: step(drift, target.drift),
                opacity: step(opacity, target.opacity),
                saturation: step(saturation, target.saturation)
            )
        }
    }

    /// How quickly the look follows a change of mood: about two thirds of
    /// the way in this many seconds, all of it in about a second.
    private static let moodTime: Double = 0.3
    /// Loudness envelope: quick to rise, slower to fall, so the orb jumps
    /// with a stressed syllable but does not flicker between words.
    private static let attack: Double = 0.05
    private static let release: Double = 0.2
    /// The most the voice grows the orb: 1.15 times at full loudness.
    private static let loudnessGrowth: Double = 0.15

    private var look: Look?
    private var loudness: Double = 0
    private var drift: Double = 0
    private var breathPhase: Double = 0
    private var lastTime: TimeInterval?

    /// Moves everything on to `time` and returns the frame to draw.
    func advance(to time: TimeInterval, mood: VoiceOrb.Mood, levels: VoiceLevelMeter?, reduceMotion: Bool) -> OrbFrame {
        // At most a tenth of a second: after a pause (the app in the
        // background, the timeline stopped) the orb carries on from where
        // it was instead of leaping.
        let dt = min(max(time - (lastTime ?? time), 0), 0.1)
        lastTime = time

        let target = Look.of(mood)
        let current = (look ?? target).moved(toward: target, by: Self.fraction(dt, Self.moodTime))
        look = current

        let heard = Self.targetLoudness(mood: mood, levels: levels)
        loudness += (heard - loudness) * Self.fraction(dt, heard > loudness ? Self.attack : Self.release)

        // Speed times time, so a new speed carries on from where the
        // clouds are instead of jumping them.
        if !reduceMotion {
            drift += dt * 2 * .pi * current.drift * (1 + 1.5 * loudness)
        }
        breathPhase += dt * 2 * .pi / current.breathPeriod
        // 0 at the bottom of a breath, 1 at the top.
        let breath = 0.5 + 0.5 * sin(breathPhase)

        if reduceMotion {
            return OrbFrame(
                scale: 1,
                drift: drift,
                opacity: current.opacity * (0.88 + 0.12 * breath),
                saturation: current.saturation,
                brightness: 0.12 * loudness
            )
        }
        return OrbFrame(
            scale: current.scale * (1 + current.breath * breath) * (1 + Self.loudnessGrowth * loudness),
            drift: drift,
            opacity: current.opacity,
            saturation: current.saturation,
            brightness: 0.05 * loudness
        )
    }

    /// The share of the remaining distance an exponential ease with time
    /// constant `timeConstant` covers in `dt`; the same at any frame rate.
    private static func fraction(_ dt: Double, _ timeConstant: Double) -> Double {
        1 - exp(-dt / timeConstant)
    }

    /// The loudness the orb follows: Pie's voice in full, the user's at a
    /// little over half, so the orb answers the user more quietly than it
    /// speaks. A muted orb ignores the room.
    private static func targetLoudness(mood: VoiceOrb.Mood, levels: VoiceLevelMeter?) -> Double {
        guard let levels else { return 0 }
        let voice = audible(levels.output)
        let user = 0.6 * audible(levels.input)
        switch mood {
        case .listening, .thinking, .speaking: return max(voice, user)
        case .muted: return voice
        case .connecting, .failed: return 0
        }
    }

    /// A meter level (0...1, where 0 is -50 dB) as loudness above the
    /// hum of a quiet room, which reads as silence.
    private static func audible(_ level: Float) -> Double {
        let floor = 0.15
        return min(max((Double(level) - floor) / (1 - floor), 0), 1)
    }
}
