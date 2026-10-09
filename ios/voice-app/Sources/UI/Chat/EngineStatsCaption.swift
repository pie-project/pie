import SwiftUI

/// What a reply cost the engine, on one line under it: "0.13 s to first
/// token · 66 tok/s · 192 of 260 prompt tokens reused". Parts the engine
/// did not report are left out.
struct EngineStatsCaption: View {
    let stats: TurnStats

    var body: some View {
        if let line = Self.line(for: stats) {
            Text(line)
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(Theme.tertiaryInk)
                .accessibilityLabel("Engine stats: \(line)")
        }
    }

    static func line(for stats: TurnStats) -> String? {
        var parts: [String] = []
        if let firstToken = stats.timeToFirstToken {
            parts.append(String(format: "%.2f s to first token", firstToken))
        }
        let rate = stats.tokensPerSecond
        if rate > 0 {
            parts.append(String(format: "%.0f tok/s", rate))
        }
        if stats.promptTokens > 0 {
            parts.append("\(stats.reused) of \(stats.promptTokens) prompt tokens reused")
        }
        if !stats.note.isEmpty {
            parts.append(stats.note)
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}
