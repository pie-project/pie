import SwiftUI

/// Shown when the engine could not boot. A failed boot is final for the
/// process, so the only way forward is to quit and open the app again.
struct EngineFailedCard: View {
    let message: String

    @EnvironmentObject private var chat: ChatController

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Pie couldn't start", systemImage: "exclamationmark.triangle.fill")
                .font(.headline)
                .foregroundStyle(Theme.red)
            Text(message)
                .font(.footnote)
                .foregroundStyle(Theme.secondaryInk)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Button {
                chat.quitToRetryEngineBoot()
            } label: {
                Text("Quit and reopen")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.onAccent)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 9)
                    .background(Theme.accentFill, in: Capsule())
            }
            .buttonStyle(.plain)
            .padding(.top, 2)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Theme.hairline, lineWidth: 0.5)
        }
    }
}
