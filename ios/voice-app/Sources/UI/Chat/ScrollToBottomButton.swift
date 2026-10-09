import SwiftUI

/// The floating arrow that appears once the end of the conversation is
/// out of sight below. The list pops it in, fades it out, and glides to
/// the end when it is tapped.
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
                .shadow(color: .black.opacity(0.08), radius: 6, y: 2)
                .frame(width: 44, height: 44)
                .contentShape(Circle())
        }
        .buttonStyle(PressScaleButtonStyle())
        .accessibilityLabel("Scroll to bottom")
    }
}
