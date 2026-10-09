import SwiftUI
import UIKit

/// The app's motion language, matched to ChatGPT's iPhone app: quiet and
/// functional. Content arrives by fading, with at most a few points of
/// rise or a slight scale; springs are critically damped, so nothing
/// overshoots; things appear a little slower than they leave. The only
/// expressive pieces are the voice orb and the streaming haptics.
///
/// Two rules every view follows:
///
/// - A state change that should animate is made inside
///   `withAnimation(Motion.…)`, at the tap or in the controller method
///   that makes it. Measured on iOS 26: the body-scoped
///   `.animation(_:body:)` modifier did not animate a change published by
///   an `ObservableObject` (the drawer snapped open in one frame), while
///   the same change inside `withAnimation` slid. `.animation(_:value:)`
///   on the view that changes is fine for view-local values.
/// - Nothing is animated per token. Streaming text is paced and faded by
///   its own renderer; a `withAnimation` per chunk would animate the whole
///   transcript's layout sixty times a second.
///
/// Reduce Motion keeps fades and drops movement and scale (`Motion.reduced`).
enum Motion {
    /// Glyph swaps, press states, small buttons appearing.
    static let control = Animation.snappy(duration: 0.2)
    /// Layout: a bubble arriving, the composer growing, the action bar, a
    /// disclosure opening.
    static let content = Animation.smooth(duration: 0.3)
    /// Something appearing on its own (an action bar, a label).
    static let fadeIn = Animation.easeOut(duration: 0.22)
    /// Something leaving. Shorter than its arrival.
    static let fadeOut = Animation.easeOut(duration: 0.14)
    /// One state replacing another in place (new chat, a label's text).
    static let crossfade = Animation.easeInOut(duration: 0.22)
    /// The drawer and other large panels.
    static let panel = Animation.smooth(duration: 0.33)
    /// A programmatic scroll (scroll-to-bottom, landing a sent message).
    static let scroll = Animation.smooth(duration: 0.38)

    /// Breathing of the reply-pending dot, one full cycle.
    static let pulsePeriod: TimeInterval = 1.2
    /// One sweep of the "Thinking" shimmer.
    static let shimmerPeriod: TimeInterval = 1.6

    /// Whether the user asked for less motion. Views should prefer
    /// `@Environment(\.accessibilityReduceMotion)`; this is for controllers.
    static var prefersReduced: Bool { UIAccessibility.isReduceMotionEnabled }

    /// `animation`, or a short fade-length ease when Reduce Motion is on,
    /// so offsets and scales jump while opacity still eases.
    static func reduced(_ animation: Animation, _ reduceMotion: Bool = prefersReduced) -> Animation {
        reduceMotion ? .easeInOut(duration: 0.18) : animation
    }
}

/// Runs `body` animated with `animation`, honouring Reduce Motion.
func withMotion<Result>(_ animation: Animation, _ body: () throws -> Result) rethrows -> Result {
    try withAnimation(Motion.reduced(animation), body)
}

// MARK: - Transitions

extension AnyTransition {
    /// Arrives fading in while rising `distance` points; leaves by fading.
    static func fadeRise(_ distance: CGFloat = 10) -> AnyTransition {
        .asymmetric(
            insertion: .opacity.combined(with: .offset(y: Motion.prefersReduced ? 0 : distance)),
            removal: .opacity
        )
    }

    /// Arrives fading in from a slightly smaller size; leaves by fading.
    static func popIn(from scale: CGFloat = 0.85) -> AnyTransition {
        .asymmetric(
            insertion: .opacity.combined(with: .scale(scale: Motion.prefersReduced ? 1 : scale)),
            removal: .opacity
        )
    }
}

// MARK: - Press feedback

/// Icon-only buttons: dim while pressed, back in a snap.
struct PressDimButtonStyle: ButtonStyle {
    var pressedOpacity: Double = 0.45

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .opacity(configuration.isPressed ? pressedOpacity : 1)
            .animation(configuration.isPressed ? nil : Motion.control, value: configuration.isPressed)
    }
}

/// Filled circle buttons and cards: shrink slightly while pressed.
struct PressScaleButtonStyle: ButtonStyle {
    var pressedScale: CGFloat = 0.93
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed && !reduceMotion ? pressedScale : 1)
            .opacity(configuration.isPressed && reduceMotion ? 0.7 : 1)
            .animation(Motion.control, value: configuration.isPressed)
    }
}

/// List rows that are buttons: the gray pressed fill ChatGPT's rows show.
struct PressHighlightButtonStyle: ButtonStyle {
    var cornerRadius: CGFloat = 10

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Color.primary.opacity(configuration.isPressed ? 0.08 : 0))
            )
            .animation(configuration.isPressed ? nil : Motion.fadeOut, value: configuration.isPressed)
    }
}

// MARK: - Haptics

/// Prepared generators for the app's haptics, so the first tap of a
/// gesture is not late. Every call takes the Settings switch, as
/// `ChatHaptics` does. Light and sparse, as in ChatGPT: a tap on send,
/// stop, copy and the like, a selection tick for toggles, and the train
/// of soft ticks while a reply streams.
@MainActor
enum Haptics {
    private static let light = UIImpactFeedbackGenerator(style: .light)
    private static let soft = UIImpactFeedbackGenerator(style: .soft)
    private static let selectionGenerator = UISelectionFeedbackGenerator()
    private static let notification = UINotificationFeedbackGenerator()
    private static var lastTick: TimeInterval = 0

    /// A discrete action: send, stop, copy, regenerate, a chip.
    static func tap(enabled: Bool) {
        guard enabled else { return }
        light.impactOccurred(intensity: 0.8)
        light.prepare()
    }

    /// A toggle or a choice: thumbs, temporary chat, the drawer settling.
    static func selection(enabled: Bool) {
        guard enabled else { return }
        selectionGenerator.selectionChanged()
        selectionGenerator.prepare()
    }

    static func failure(enabled: Bool) {
        guard enabled else { return }
        notification.notificationOccurred(.error)
    }

    /// Call when a reply starts streaming, so the first tick is on time.
    static func prepareStreaming(enabled: Bool) {
        guard enabled else { return }
        soft.prepare()
    }

    /// One tick of the streaming train, at most every 70 ms: called each
    /// time new words are revealed, it reads as the phone typing.
    static func streamTick(enabled: Bool) {
        guard enabled else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastTick >= 0.07 else { return }
        lastTick = now
        soft.impactOccurred(intensity: 0.5)
        soft.prepare()
    }
}
