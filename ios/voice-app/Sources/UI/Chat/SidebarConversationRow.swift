import SwiftUI

/// A saved conversation in the sidebar: its title on one line.
///
/// Pressed, it shows ChatGPT's gray fill; the open chat keeps a tinted
/// rounded highlight, which fades from row to row rather than jumping. A
/// new title (generated, or renamed) crossfades in place.
struct SidebarConversationRow: View {
    let conversation: Conversation
    let isOpen: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(conversation.displayTitle)
                .font(.body)
                .foregroundStyle(Theme.ink)
                .lineLimit(1)
                .contentTransition(.opacity)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .contentShape(Rectangle())
        }
        .buttonStyle(PressHighlightButtonStyle())
        // Behind the pressed fill, so pressing the open row darkens it.
        .background {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Theme.accentWash)
                .opacity(isOpen ? 1 : 0)
        }
        .animation(Motion.crossfade, value: isOpen)
        .accessibilityAddTraits(isOpen ? .isSelected : [])
    }
}
