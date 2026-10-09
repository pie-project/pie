import SwiftUI

/// The composer's round buttons: a 36 pt circle inside a 44 pt target.
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
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: glyphSize, weight: style == .filled ? .bold : .regular))
                .foregroundStyle(foreground)
                .frame(width: 36, height: 36)
                .background { background }
                .frame(width: 44, height: 44)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .accessibilityLabel(label)
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
        case .filled: return isEnabled ? Theme.onAccent : Theme.tertiaryInk
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
            Circle().fill(isEnabled ? Theme.accentFill : Theme.surfaceStrong)
        }
    }
}
