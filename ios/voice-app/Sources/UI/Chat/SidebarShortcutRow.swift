import SwiftUI

/// One of the sidebar's fixed rows above the chat history.
struct SidebarShortcutRow<Icon: View>: View {
    let title: String
    @ViewBuilder let icon: () -> Icon
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                icon()
                    .font(.system(size: 18))
                    .frame(width: 24)
                Text(title)
                    .font(.body.weight(.medium))
                Spacer(minLength: 0)
            }
            .foregroundStyle(Theme.ink)
            .padding(.vertical, 10)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
