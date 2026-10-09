import SwiftUI

/// The composer's round buttons: a 36 pt circle inside a 44 pt target.
///
/// A button that changes role keeps its circle and swaps only its glyph,
/// as ChatGPT's does (voice to send to stop, waveform to the dictation
/// checkmark): the old glyph shrinks away while the new one grows in, both
/// at once, so the circle is never empty, and the circle stays exactly
/// where it is.
///
/// The swap is our own (`GlyphSwap`), not SF Symbols' "replace" content
/// transition. Recorded in the iOS 26 Simulator, the replace ran on its
/// own fixed ~0.4 s timing whatever animation we gave it, snapped
/// waveform to checkmark in one frame inside dictation's opening
/// animation, and with the keyboard leaving on send it dropped the arrow
/// at once and left the circle empty for ~0.18 s before the stop square
/// grew.
struct ComposerIconButton: View {
    enum Style {
        /// A bare glyph: dictation, cancel.
        case plain
        /// A hairline circle: "+".
        case outlined
        /// A filled accent circle with a white glyph: send, stop, voice mode.
        case filled
    }

    let symbol: String
    let style: Style
    let label: String
    var isEnabled = true
    /// A spinner in place of the glyph, while something finishes (the
    /// dictation's last words being transcribed).
    var showsProgress = false
    let action: () -> Void

    var body: some View {
        pressStyled(Button(action: action) { face })
            .disabled(!isEnabled)
            // Disabled send is the same circle, faded, as in ChatGPT, and
            // it eases back when sending becomes possible.
            .opacity(isEnabled ? 1 : 0.35)
            .animation(Motion.control, value: isEnabled)
            .accessibilityLabel(label)
    }

    private var face: some View {
        ZStack {
            background
            GlyphSwap(symbol: symbol, style: style, color: foreground)
                .opacity(showsProgress ? 0 : 1)
            // Only the filled circle ever shows a spinner. Always there and
            // faded rather than inserted, for the reason `GlyphSwap` gives.
            if style == .filled {
                ProgressView()
                    .controlSize(.small)
                    .tint(foreground)
                    .opacity(showsProgress ? 1 : 0)
                    // The button's label already says what it does.
                    .accessibilityHidden(true)
            }
        }
        .frame(width: 36, height: 36)
        .frame(width: 44, height: 44)
        .contentShape(Circle())
        // Local to this button, so nothing else on screen takes part.
        .animation(Motion.control, value: showsProgress)
    }

    /// Filled circles shrink a little while pressed; bare and outlined
    /// icons dim, as ChatGPT's do.
    @ViewBuilder
    private func pressStyled(_ button: Button<some View>) -> some View {
        if style == .filled {
            button.buttonStyle(PressScaleButtonStyle())
        } else {
            button.buttonStyle(PressDimButtonStyle())
        }
    }

    fileprivate static func glyphFont(_ symbol: String, _ style: Style) -> Font {
        switch style {
        case .plain, .outlined: return .system(size: 18, weight: .regular)
        case .filled: return .system(size: symbol == "stop.fill" ? 12 : 15, weight: .bold)
        }
    }

    private var foreground: Color {
        switch style {
        case .plain, .outlined: return Theme.ink
        case .filled: return Theme.onAccent
        }
    }

    @ViewBuilder
    private var background: some View {
        switch style {
        case .plain:
            Color.clear
        case .outlined:
            Circle().strokeBorder(Theme.hairline, lineWidth: 1)
        case .filled:
            Circle().fill(Theme.accentFill)
        }
    }
}

/// A glyph that changes symbol by shrinking the old one away while the new
/// one grows in, both at once: the old one shrinks to half size and fades
/// in 0.14 s, the new one grows from half size in 0.2 s. So the circle
/// always holds a glyph, and the old one is nearly gone before the new one
/// is legible, so the two never read as one shape. Reduce Motion keeps the
/// fades only.
///
/// Two glyph layers are always there and trade places, so nothing is
/// inserted or removed. Recorded with `.id(symbol)` and a removal
/// transition, the old glyph vanished in one frame in every case (typing,
/// send, dictation, even inside an explicit `withAnimation`) while the
/// new one did grow in: inside this button's label the removal never ran.
/// Opacity and scale changes on views that stay do animate there.
private struct GlyphSwap: View {
    let symbol: String
    let style: ComposerIconButton.Style
    let color: Color

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// The symbol each layer draws. The hidden layer takes the next one.
    @State private var layers: [String]
    /// The layer showing `symbol`.
    @State private var front = 0

    init(symbol: String, style: ComposerIconButton.Style, color: Color) {
        self.symbol = symbol
        self.style = style
        self.color = color
        _layers = State(initialValue: [symbol, symbol])
    }

    var body: some View {
        ZStack {
            ForEach(0..<2, id: \.self) { layer in
                let isFront = layer == front
                Image(systemName: layers[layer])
                    .font(ComposerIconButton.glyphFont(layers[layer], style))
                    .foregroundStyle(color)
                    // The hidden layer changes symbol while invisible; no
                    // symbol effect of its own on top of the swap.
                    .contentTransition(.identity)
                    .opacity(isFront ? 1 : 0)
                    .scaleEffect(isFront || reduceMotion ? 1 : 0.5)
                    // Arriving takes 0.2 s, leaving 0.14 s.
                    .animation(isFront ? Motion.control : Motion.fadeOut, value: front)
            }
        }
        .onChange(of: symbol) { _, new in
            // Our own transaction, whatever the role change came in (a
            // keystroke, the send's, dictation's crossfade): the swap runs
            // on this timing every time.
            withAnimation(Motion.control) {
                let hidden = 1 - front
                layers[hidden] = new
                front = hidden
            }
        }
    }
}
