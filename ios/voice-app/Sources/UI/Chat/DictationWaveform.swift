import SwiftUI

/// The microphone's loudness as a strip of thin rounded bars scrolling in
/// from the right, drawn over the text field while dictating, as in
/// ChatGPT: each bar records how loud the voice was when it entered, and
/// silence leaves a row of flat dots. The left edge fades out.
///
/// A `TimelineView` redraws it every display frame, and `meter` is moved on
/// to that frame's time before drawing, so the strip glides continuously
/// rather than stepping once per microphone report. The timeline is paused
/// whenever the strip should hold still, so it costs nothing then.
struct DictationWaveform: View {
    let meter: DictationLevelMeter
    /// Holds the strip still and dims it: the microphone has stopped on its
    /// own, or the last words are being transcribed.
    var isPaused = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private static let barWidth: CGFloat = 2.5
    private static let gap: CGFloat = 2.5
    /// Silence: a dot.
    private static let minHeight: CGFloat = 2.5
    private static let maxHeight: CGFloat = 24
    private static let edgeFade: CGFloat = 16

    var body: some View {
        TimelineView(.animation(minimumInterval: nil, paused: isPaused)) { timeline in
            let _ = meter.advance(to: timeline.date.timeIntervalSinceReferenceDate)
            // Copied out before drawing, so the canvas draws one consistent frame.
            let bars = meter.bars
            let live = meter.live
            let travel = meter.travel
            Canvas { context, size in
                if reduceMotion {
                    Self.drawStill(in: &context, size: size, level: live)
                } else {
                    Self.drawScrolling(in: &context, size: size, bars: bars, live: live, travel: travel)
                }
            }
        }
        .frame(maxWidth: .infinity, minHeight: 44, maxHeight: 44)
        .mask {
            HStack(spacing: 0) {
                LinearGradient(colors: [.clear, .black], startPoint: .leading, endPoint: .trailing)
                    .frame(width: Self.edgeFade)
                Rectangle()
            }
        }
        .opacity(isPaused ? 0.4 : 1)
        .animation(Motion.crossfade, value: isPaused)
        .accessibilityHidden(true)
    }

    /// The bar being recorded enters at the right edge; each recorded bar
    /// sits one spacing further left, all shifted by `travel`.
    private static func drawScrolling(
        in context: inout GraphicsContext,
        size: CGSize,
        bars: [CGFloat],
        live: CGFloat,
        travel: CGFloat
    ) {
        let pitch = barWidth + gap
        let middle = size.height / 2
        // Position 0 is the live bar, 1 the newest recorded one, and so on.
        for position in 0...bars.count {
            let x = size.width - barWidth - (CGFloat(position) + travel) * pitch
            if x < -barWidth { break }
            let level = position == 0 ? live : bars[bars.count - position]
            let height = minHeight + level * (maxHeight - minHeight)
            let bar = CGRect(x: x, y: middle - height / 2, width: barWidth, height: height)
            context.fill(Path(roundedRect: bar, cornerRadius: barWidth / 2), with: .color(Theme.ink))
        }
    }

    /// Reduce Motion: no scrolling and no growing bars. A still row of dots
    /// that brightens with the voice still shows the microphone hears.
    private static func drawStill(in context: inout GraphicsContext, size: CGSize, level: CGFloat) {
        let pitch = barWidth + gap
        let middle = size.height / 2
        let color = Theme.ink.opacity(0.35 + 0.65 * Double(level))
        var x = size.width - barWidth
        while x > -barWidth {
            let dot = CGRect(x: x, y: middle - minHeight / 2, width: barWidth, height: minHeight)
            context.fill(Path(roundedRect: dot, cornerRadius: barWidth / 2), with: .color(color))
            x -= pitch
        }
    }
}
