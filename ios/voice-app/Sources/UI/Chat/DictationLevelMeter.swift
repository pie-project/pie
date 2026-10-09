import CoreGraphics
import Foundation

/// The dictation microphone's recent loudness, as the bars the waveform
/// draws.
///
/// Deliberately not an `ObservableObject`. The microphone reports its
/// level about 30 times a second, and publishing each report re-rendered
/// the whole composer and moved the strip in 5 pt steps. Instead the
/// controller drops reports in here, and the waveform reads this once per
/// display frame (`DictationWaveform`'s `TimelineView`), so the strip
/// scrolls at the screen's rate however unevenly the reports arrive.
final class DictationLevelMeter {
    /// One bar is recorded this often; between two, the strip glides one
    /// bar's spacing to the left.
    static let sampleInterval: TimeInterval = 0.06
    /// More bars than fit across the widest composer, so the strip is
    /// always full, flat dots on the left before anything was said.
    static let barCount = 90

    /// Rises to a louder level in about 30 ms and falls back in about
    /// 120 ms, so syllables snap up and decay as ChatGPT's bars do.
    private static let attack: TimeInterval = 0.03
    private static let release: TimeInterval = 0.12

    /// Recorded bars, oldest first, each 0...1. A bar keeps the loudness
    /// it was recorded with as it scrolls away.
    private(set) var bars = Array(repeating: CGFloat(0), count: barCount)
    /// The loudness now, smoothed: the bar entering at the right edge.
    private(set) var live: CGFloat = 0
    /// How far the strip has glided since the last bar was recorded, as a
    /// fraction (0..<1) of one bar's spacing.
    private(set) var travel: CGFloat = 0

    /// The last level the microphone reported.
    private var target: CGFloat = 0
    private var lastFrame: TimeInterval?
    private var lastRecorded: TimeInterval = 0

    /// A level from the microphone, 0...1.
    func report(_ level: Float) {
        target = CGFloat(min(max(level, 0), 1))
    }

    func reset() {
        bars = Array(repeating: 0, count: Self.barCount)
        live = 0
        travel = 0
        target = 0
        lastFrame = nil
    }

    /// Brings the strip up to `now` (seconds): eases `live` toward the last
    /// report and records a bar every `sampleInterval`. Asking twice for
    /// the same moment changes nothing.
    func advance(to now: TimeInterval) {
        guard let last = lastFrame else {
            lastFrame = now
            lastRecorded = now
            return
        }
        let elapsed = now - last
        guard elapsed > 0 else { return }
        lastFrame = now

        // One-pole smoothing, frame-rate independent: after `tau` seconds
        // `live` has covered about 63 % of the way to the target.
        let tau = target > live ? Self.attack : Self.release
        live += (target - live) * CGFloat(1 - exp(-elapsed / tau))

        // After a stall (the app in the background), start afresh rather
        // than replaying a burst of identical bars.
        if now - lastRecorded > 1 { lastRecorded = now }
        while now - lastRecorded >= Self.sampleInterval {
            bars.removeFirst()
            bars.append(live)
            lastRecorded += Self.sampleInterval
        }
        travel = CGFloat((now - lastRecorded) / Self.sampleInterval)
    }
}
