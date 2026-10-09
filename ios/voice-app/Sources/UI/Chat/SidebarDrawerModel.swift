import QuartzCore
import SwiftUI

/// Where the sidebar drawer is on screen, from 0 (shut) to 1 (open), and
/// the spring that moves it.
///
/// This one number is the drawer's position at every moment. The finger
/// writes it while dragging; otherwise a display-link spring writes it,
/// frame by frame. Because what is on screen is always this value, a
/// drawer that is still sliding can be caught wherever it is, and a fling
/// hands its speed to the spring, as in ChatGPT. (A SwiftUI animation
/// would hold only the destination; grabbing the drawer mid-flight would
/// make it jump back to wherever the model said it was.)
///
/// The spring is critically damped: it gets there as fast as it can
/// without passing the end, so the window behind the drawer never shows.
///
/// Only the drawer's layers observe it, so only they redraw on each frame
/// of a slide; `RootView` holds it without observing it.
@MainActor
final class SidebarDrawerModel: ObservableObject {
    /// 0 = shut, 1 = open; what the screen shows right now.
    @Published private(set) var position: CGFloat = 0

    /// Called each time the drawer arrives at the shut position, so a
    /// presentation waiting on the close (`AppRouter.closeSidebar(then:)`)
    /// can start.
    var onClosed: (() -> Void)?

    /// The drawer is following a finger.
    var isDragging: Bool { dragOrigin != nil }

    /// Where the drawer was when the finger took hold of it.
    private var dragOrigin: CGFloat?
    private var flight: Flight?
    /// The drawer's speed on the last frame of a slide, in drawer widths
    /// per second.
    private var speed: CGFloat = 0
    private var displayLink: CADisplayLink?

    /// The spring's stiffness as an angular frequency: a response of
    /// 0.33 s, the time it takes to cover most of the distance, as in
    /// `Motion.panel` (`.smooth(duration: 0.33)`).
    private static let omega = 2 * Double.pi / 0.33
    /// Reduce Motion: a plain ease, no spring, for the button paths.
    private static let reducedDuration: Double = 0.25
    /// A release faster than this (points per second) goes where it was
    /// flung, however far the drawer had come.
    private static let flingSpeed: CGFloat = 400

    // MARK: - Driving it

    /// Slides open or shut on its own (a button, a tap on the dimmed chat,
    /// the tour). `velocity` is in drawer widths per second; left out, the
    /// drawer keeps the speed it already has, so turning it round mid-slide
    /// (a tap on the dim while it opens) reverses it smoothly rather than
    /// stopping it dead first.
    ///
    /// Already heading there, it carries on: the speed a fling gave it is
    /// kept rather than restarted from rest.
    func move(toOpen open: Bool, velocity: CGFloat? = nil, reduceMotion: Bool = Motion.prefersReduced) {
        let target: CGFloat = open ? 1 : 0
        if let flight, flight.target == target { return }
        let velocity = velocity ?? (flight == nil ? 0 : speed)
        dragOrigin = nil
        guard position != target else {
            stop()
            if target == 0 { onClosed?() }
            return
        }
        flight = Flight(
            target: target,
            start: CACurrentMediaTime(),
            from: position,
            velocity: Self.velocityWithoutOvershoot(velocity, from: position, to: target),
            omega: Self.omega,
            easeDuration: reduceMotion ? Self.reducedDuration : nil
        )
        startDisplayLink()
    }

    /// The finger took hold: the drawer stops where it is and follows the
    /// finger from there.
    func beginDrag() {
        stop()
        dragOrigin = position
    }

    /// `distance` is the finger's sideways travel since `beginDrag()`, in
    /// points; `width` is the sidebar's.
    func drag(by distance: CGFloat, width: CGFloat) {
        guard let dragOrigin, width > 0 else { return }
        position = min(max(dragOrigin + distance / width, 0), 1)
    }

    /// The finger let go, moving at `velocity` points per second. A fling
    /// goes where it was thrown; a slow release settles on the nearer
    /// side. Returns whether the drawer is now heading open.
    @discardableResult
    func endDrag(velocity: CGFloat, width: CGFloat, reduceMotion: Bool = Motion.prefersReduced) -> Bool {
        guard isDragging else { return (flight?.target ?? position) > 0.5 }
        let open = abs(velocity) > Self.flingSpeed ? velocity > 0 : position > 0.5
        dragOrigin = nil
        move(toOpen: open, velocity: width > 0 ? velocity / width : 0, reduceMotion: reduceMotion)
        return open
    }

    // MARK: - The spring

    /// One slide to `target`, either the spring or Reduce Motion's ease.
    private struct Flight {
        let target: CGFloat
        let start: CFTimeInterval
        let from: CGFloat
        /// Drawer widths per second at `start` (the spring only).
        let velocity: CGFloat
        /// The spring's stiffness (`SidebarDrawerModel.omega`).
        let omega: Double
        /// Set for Reduce Motion's plain ease instead of the spring.
        let easeDuration: Double?
        /// `onClosed` has been called for this slide.
        var hasReportedClose = false

        /// Where the drawer is `elapsed` seconds in, how fast it is going
        /// (drawer widths per second), and whether it has arrived.
        func position(after elapsed: Double) -> (value: CGFloat, speed: CGFloat, arrived: Bool) {
            let t = max(0, elapsed)
            if let easeDuration {
                let progress = min(1, t / easeDuration)
                // Ease in and out (cubic), as `.easeInOut`.
                let eased = progress < 0.5
                    ? 4 * progress * progress * progress
                    : 1 - pow(-2 * progress + 2, 3) / 2
                return (from + (target - from) * CGFloat(eased), 0, progress >= 1)
            }
            // A critically damped spring, solved exactly:
            //   x(t) = (x0 + (v0 + w x0) t) e^(-w t), x = distance from target.
            let x0 = Double(from - target)
            let v0 = Double(velocity)
            let b = v0 + omega * x0
            let decay = exp(-omega * t)
            let offset = (x0 + b * t) * decay
            let speed = (v0 - omega * b * t) * decay
            // Half a point or so on a phone's sidebar, and barely moving.
            let arrived = abs(offset) < 0.0015 && abs(speed) < 0.05
            return (target + CGFloat(offset), CGFloat(speed), arrived)
        }
    }

    /// A critically damped spring released faster than `omega` times its
    /// distance would pass the target before settling. Such a fling is
    /// capped at the fastest speed that still arrives without overshoot
    /// (it then simply decays into place); slower ones keep their speed.
    private static func velocityWithoutOvershoot(_ velocity: CGFloat, from: CGFloat, to target: CGFloat) -> CGFloat {
        let distance = target - from
        guard distance != 0, velocity * distance > 0 else { return velocity }
        let limit = CGFloat(omega) * abs(distance)
        return abs(velocity) > limit ? limit * (distance > 0 ? 1 : -1) : velocity
    }

    fileprivate func step(_ link: CADisplayLink) {
        guard let flight else {
            stop()
            return
        }
        // `targetTimestamp` is when this frame will be on screen, so the
        // drawer is drawn where it should be at that moment.
        let (value, speed, arrived) = flight.position(after: link.targetTimestamp - flight.start)
        self.speed = speed
        position = arrived ? flight.target : min(max(value, 0), 1)
        // Shut as far as anyone can see (within about 3 points): whatever
        // waits for the close starts now, not after the spring's last,
        // invisible creep.
        if flight.target == 0, position < 0.01, !flight.hasReportedClose {
            self.flight?.hasReportedClose = true
            onClosed?()
        }
        if arrived { stop() }
    }

    private func startDisplayLink() {
        guard displayLink == nil else { return }
        let link = CADisplayLink(target: DisplayLinkTarget(self), selector: #selector(DisplayLinkTarget.tick(_:)))
        // ProMotion phones draw the slide at 120 Hz.
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120, preferred: 120)
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    private func stop() {
        flight = nil
        speed = 0
        displayLink?.invalidate()
        displayLink = nil
    }
}

/// The display link's target. A display link keeps its target alive, so
/// it holds this small object rather than the model; the model is held
/// weakly and the link is invalidated once the slide ends.
@MainActor
private final class DisplayLinkTarget: NSObject {
    private weak var model: SidebarDrawerModel?

    init(_ model: SidebarDrawerModel) {
        self.model = model
    }

    @objc func tick(_ link: CADisplayLink) {
        guard let model else {
            link.invalidate()
            return
        }
        model.step(link)
    }
}
