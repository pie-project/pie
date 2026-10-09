import SwiftUI

/// The floating arrow that appears once the user has scrolled up from the
/// newest message.
struct ScrollToBottomButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "arrow.down")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Theme.ink)
                .frame(width: 36, height: 36)
                .background(Theme.background, in: Circle())
                .overlay { Circle().strokeBorder(Theme.hairline, lineWidth: 1) }
                .frame(width: 44, height: 44)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Scroll to bottom")
    }
}
