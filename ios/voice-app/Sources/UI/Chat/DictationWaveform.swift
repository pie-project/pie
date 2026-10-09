import SwiftUI

/// The microphone level as a strip of bars scrolling in from the right,
/// shown in place of the text field while dictating.
struct DictationWaveform: View {
    let level: Float

    @State private var samples = Array(repeating: CGFloat(0), count: 56)

    var body: some View {
        HStack(spacing: 2.5) {
            ForEach(samples.indices, id: \.self) { index in
                Capsule()
                    .fill(Theme.ink)
                    .frame(width: 2.5, height: 3 + samples[index] * 26)
            }
        }
        // Wider than the space it gets on purpose: the oldest bars are cut
        // off at the left edge as new ones arrive at the right.
        .frame(maxWidth: .infinity, minHeight: 44, maxHeight: 44, alignment: .trailing)
        .clipped()
        .onChange(of: level) { _, newLevel in
            samples.removeFirst()
            samples.append(CGFloat(min(max(newLevel, 0), 1)))
        }
        .accessibilityHidden(true)
    }
}
