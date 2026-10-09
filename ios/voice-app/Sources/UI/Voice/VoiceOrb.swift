import SwiftUI

/// The sphere at the centre of voice mode: a Yale-blue orb that breathes
/// while it listens, shimmers while the model thinks, and pulses with the
/// voice while it speaks.
struct VoiceOrb: View {

    enum Mood: Equatable {
        case idle
        case listening
        case thinking
        case speaking
        /// The microphone is off; the orb goes grey.
        case muted
        case failed
    }

    let mood: Mood
    /// 0...1, from the microphone.
    var inputLevel: Float = 0
    /// 0...1, from what is being spoken.
    var outputLevel: Float = 0
    var diameter: CGFloat = 240

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.animation(minimumInterval: nil, paused: isStill)) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            ZStack {
                if mood == .speaking && !reduceMotion {
                    ripples(at: t)
                }
                sphere(at: t)
                    .scaleEffect(reduceMotion ? 1 : breathing(at: t))
            }
        }
        // The level arrives 20-30 times a second; springing towards it
        // turns those steps into a smooth swell.
        .scaleEffect(levelScale)
        .animation(.interactiveSpring(response: 0.22, dampingFraction: 0.72), value: levelScale)
        .animation(.easeInOut(duration: 0.45), value: mood)
        .frame(width: diameter, height: diameter)
    }

    // MARK: - Layers

    private func sphere(at t: TimeInterval) -> some View {
        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [Theme.yaleMedium, Theme.yale],
                        center: UnitPoint(x: 0.42, y: 0.38),
                        startRadius: 0,
                        endRadius: diameter * 0.62
                    )
                )

            currents(at: t)

            shimmer(at: t)
                .opacity(mood == .thinking ? 1 : 0)

            // A soft highlight, so the disc reads as a sphere.
            Circle()
                .fill(
                    RadialGradient(
                        colors: [Color.white.opacity(0.32), Color.white.opacity(0)],
                        center: UnitPoint(x: 0.34, y: 0.28),
                        startRadius: 0,
                        endRadius: diameter * 0.42
                    )
                )

            Circle()
                .fill(
                    RadialGradient(
                        colors: [Theme.surfaceStrong, Theme.tertiaryInk],
                        center: UnitPoint(x: 0.42, y: 0.38),
                        startRadius: 0,
                        endRadius: diameter * 0.7
                    )
                )
                .opacity(mood == .muted ? 1 : 0)

            Circle()
                .fill(
                    RadialGradient(
                        colors: [Theme.red.opacity(0.7), Theme.red],
                        center: UnitPoint(x: 0.42, y: 0.38),
                        startRadius: 0,
                        endRadius: diameter * 0.62
                    )
                )
                .opacity(mood == .failed ? 1 : 0)
        }
        .clipShape(Circle())
        .shadow(color: glowColor.opacity(0.45), radius: diameter * 0.15)
    }

    /// Two soft lights drifting inside the sphere, so it looks alive even
    /// when nothing else moves. They speed up while the model thinks.
    private func currents(at t: TimeInterval) -> some View {
        let speed = mood == .thinking ? 0.9 : 0.35
        let angle = t * speed
        let r = diameter * 0.18
        return ZStack {
            Ellipse()
                .fill(Theme.yaleLight.opacity(0.45))
                .frame(width: diameter * 0.55, height: diameter * 0.38)
                .offset(x: cos(angle) * r, y: sin(angle) * r)
            Ellipse()
                .fill(Theme.yaleMedium.opacity(0.55))
                .frame(width: diameter * 0.5, height: diameter * 0.42)
                .offset(x: cos(angle * 1.3 + .pi) * r, y: sin(angle * 1.3 + .pi) * r * 0.8)
        }
        .blur(radius: diameter * 0.09)
        .opacity(mood == .muted || mood == .failed ? 0 : 1)
    }

    /// A band of light sweeping around the rim while the model thinks.
    private func shimmer(at t: TimeInterval) -> some View {
        Circle()
            .fill(
                AngularGradient(
                    colors: [
                        .clear,
                        Color.white.opacity(0.38),
                        .clear,
                        .clear,
                        Theme.yaleLight.opacity(0.5),
                        .clear,
                    ],
                    center: .center,
                    angle: .radians(t * 1.6)
                )
            )
            .blur(radius: diameter * 0.05)
    }

    /// Rings spreading out from the sphere, stronger the louder the voice.
    private func ripples(at t: TimeInterval) -> some View {
        let strength = 0.12 + 0.6 * Double(min(max(outputLevel, 0), 1))
        return ZStack {
            ForEach(0..<3, id: \.self) { index in
                let phase = (t / 1.8 + Double(index) / 3).truncatingRemainder(dividingBy: 1)
                Circle()
                    .stroke(Theme.yaleMedium, lineWidth: 2)
                    .frame(width: diameter, height: diameter)
                    .scaleEffect(1 + 0.38 * phase)
                    .opacity((1 - phase) * strength)
            }
        }
    }

    // MARK: - Motion

    private var isStill: Bool {
        reduceMotion || mood == .muted || mood == .failed
    }

    /// Slow breathing, a few percent either way.
    private func breathing(at t: TimeInterval) -> CGFloat {
        switch mood {
        case .listening: return 1 + 0.025 * CGFloat(sin(t * 2 * .pi / 4))
        case .idle: return 1 + 0.012 * CGFloat(sin(t * 2 * .pi / 5))
        case .thinking: return 1 + 0.015 * CGFloat(sin(t * 2 * .pi / 2.2))
        case .speaking, .muted, .failed: return 1
        }
    }

    private var levelScale: CGFloat {
        switch mood {
        case .listening: return 1 + 0.12 * CGFloat(min(max(inputLevel, 0) * 1.4, 1))
        case .speaking: return 1 + 0.12 * CGFloat(min(max(outputLevel, 0), 1))
        case .idle, .thinking, .muted, .failed: return 1
        }
    }

    private var glowColor: Color {
        switch mood {
        case .muted: return .clear
        case .failed: return Theme.red
        case .idle, .listening, .thinking, .speaking: return Theme.yaleMedium
        }
    }
}
