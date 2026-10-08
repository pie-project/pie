import SwiftUI

struct TranscriptView: View {

    let turns: [Turn]
    let partialTranscript: String
    let isListening: Bool

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if turns.isEmpty && partialTranscript.isEmpty {
                        EmptyStatePrompt()
                    }

                    ForEach(turns) { turn in
                        TurnBubble(turn: turn)
                            .id(turn.id)
                    }

                    if isListening && !partialTranscript.isEmpty {
                        PartialBubble(text: partialTranscript)
                            .id(Self.partialID)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }
            .onChange(of: turns.count) { scroll(proxy) }
            .onChange(of: turns.last?.text ?? "") { scroll(proxy) }
            .onChange(of: partialTranscript) { scroll(proxy) }
        }
    }

    private static let partialID = "partial"

    private func scroll(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.2)) {
            if isListening && !partialTranscript.isEmpty {
                proxy.scrollTo(Self.partialID, anchor: .bottom)
            } else if let last = turns.last {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        }
    }
}

private struct EmptyStatePrompt: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Press the microphone and ask something.")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("Speech recognition, the language model, and the KV cache all stay on the device. Nothing is sent anywhere.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 24)
    }
}

private struct TurnBubble: View {

    let turn: Turn

    var body: some View {
        HStack {
            if turn.speaker == .user { Spacer(minLength: 40) }

            VStack(alignment: turn.speaker == .user ? .trailing : .leading, spacing: 6) {
                Text(displayText)
                    .font(.body)
                    .foregroundStyle(turn.speaker == .user ? Color.white : Color.primary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(background)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .textSelection(.enabled)

                if let stats = turn.stats {
                    StatsChip(stats: stats)
                }
            }

            if turn.speaker == .assistant { Spacer(minLength: 40) }
        }
    }

    /// A caret while streaming, so an empty assistant bubble doesn't read
    /// as a hang during the first token.
    private var displayText: String {
        if turn.isStreaming {
            return turn.text.isEmpty ? "…" : turn.text + "▍"
        }
        return turn.text
    }

    private var background: Color {
        turn.speaker == .user
            ? Color.accentColor
            : Color(.secondarySystemBackground)
    }
}

private struct PartialBubble: View {

    let text: String

    var body: some View {
        HStack {
            Spacer(minLength: 40)
            Text(text)
                .font(.body)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(Color.accentColor.opacity(0.4), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                )
        }
    }
}

/// Per-turn KV accounting. This is the number that distinguishes running
/// on Pie from running any other on-device chat stack: on turn two and
/// beyond, the conversation so far is not prefilled again.
private struct StatsChip: View {

    let stats: TurnStats

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                if stats.resumed && stats.reused > 0 {
                    Label("\(stats.reused) KV reused", systemImage: "arrow.triangle.2.circlepath")
                }
                Text("prefill \(stats.newPrefill)")
                Text("· \(stats.generated) tok")
                if stats.tokensPerSecond > 0 {
                    Text("· \(stats.tokensPerSecond, specifier: "%.1f") tok/s")
                }
            }
            if !stats.note.isEmpty {
                Text(stats.note)
                    .foregroundStyle(.orange)
            }
        }
        .font(.caption2)
        .foregroundStyle(.tertiary)
        .padding(.horizontal, 4)
    }
}
