import SwiftUI

/// A saved conversation in the sidebar: its title on one line.
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
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 10)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isOpen ? .isSelected : [])
    }
}
