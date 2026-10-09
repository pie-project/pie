import SwiftUI

/// ChatGPT's temporary-chat symbol: a speech bubble with a dashed outline,
/// filled while the open chat is temporary. SF Symbols has no dashed
/// bubble, so it is drawn.
///
/// Turning it on or off swaps the two drawings the way an SF Symbol's
/// "replace" effect does: the old one shrinks away as the new one grows
/// in. With Reduce Motion they only crossfade.
struct TemporaryChatGlyph: View {
    var isFilled = false
    var size: CGFloat = 20

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            if isFilled {
                SpeechBubbleShape()
                    .fill()
                    .transition(replace)
            } else {
                SpeechBubbleShape()
                    .stroke(style: StrokeStyle(lineWidth: size * 0.085, lineCap: .round, dash: [size * 0.12, size * 0.13]))
                    .transition(replace)
            }
        }
        .frame(width: size, height: size)
        .animation(Motion.control, value: isFilled)
    }

    private var replace: AnyTransition {
        reduceMotion ? .opacity : .scale(scale: 0.5).combined(with: .opacity)
    }
}

/// A round bubble with its tail at the lower left, like `bubble.left`.
private struct SpeechBubbleShape: Shape {
    func path(in rect: CGRect) -> Path {
        let radius = min(rect.width, rect.height) * 0.44
        let center = CGPoint(x: rect.midX, y: rect.minY + rect.height * 0.46)
        let tip = CGPoint(x: rect.minX + rect.width * 0.07, y: rect.maxY - rect.height * 0.05)
        func point(_ degrees: CGFloat) -> CGPoint {
            let radians = degrees * .pi / 180
            return CGPoint(x: center.x + radius * cos(radians), y: center.y + radius * sin(radians))
        }
        var path = Path()
        path.move(to: tip)
        path.addLine(to: point(162))
        // Angles grow clockwise on screen (y points down), so this runs
        // over the top and round to the bottom, leaving the tail's gap.
        path.addArc(center: center, radius: radius, startAngle: .degrees(162), endAngle: .degrees(112), clockwise: false)
        path.closeSubpath()
        return path
    }
}
