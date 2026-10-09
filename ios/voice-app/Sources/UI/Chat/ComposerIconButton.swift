import SwiftUI

/// The composer's round buttons: a 36 pt circle inside a 44 pt target.
///
/// A button that changes role keeps its circle and swaps only its glyph,
/// as ChatGPT's does (voice to send to stop): the new symbol replaces the
/// old one with the system's symbol "replace" effect, a quick shrink and
/// grow, while the circle stays exactly where it is.
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
            if showsProgress {
                ProgressView()
                    .controlSize(.small)
                    .tint(foreground)
                    .transition(.opacity)
            } else {
                Image(systemName: symbol)
                    .font(.system(size: glyphSize, weight: style == .filled ? .bold : .regular))
                    .foregroundStyle(foreground)
                    // Down-up names the classic replace (old glyph shrinks
                    // away, new one grows in). Plain `.replace` on iOS 26
                    // draws the checkmark on stroke by stroke, and in the
                    // Simulator it stalled half drawn for half a second.
                    .contentTransition(.symbolEffect(.replace.downUp))
                    .transition(.opacity)
            }
        }
        .frame(width: 36, height: 36)
        .frame(width: 44, height: 44)
        .contentShape(Circle())
        // Local to this button: only the glyph changes, so nothing else
        // on screen takes part in these animations.
        .animation(Motion.control, value: symbol)
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

    private var glyphSize: CGFloat {
        switch style {
        case .plain, .outlined: return 18
        case .filled: return symbol == "stop.fill" ? 12 : 15
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
