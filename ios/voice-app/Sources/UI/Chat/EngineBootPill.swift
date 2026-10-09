import SwiftUI

/// The slim status under the top bar while the model loads.
struct EngineBootPill: View {
    let seconds: Int

    var body: some View {
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
                .tint(Theme.accent)
            Text("Loading \(BootedModel.current.label) on this iPhone\u{2026} \(seconds)s")
                .font(.footnote.weight(.medium))
                .monospacedDigit()
                .foregroundStyle(Theme.secondaryInk)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .background(Theme.surface, in: Capsule())
        .overlay { Capsule().strokeBorder(Theme.hairline, lineWidth: 0.5) }
        .accessibilityElement(children: .combine)
    }
}
