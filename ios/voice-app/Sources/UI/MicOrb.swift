import SwiftUI

/// The one control that matters: press to talk, press again to interrupt.
///
/// The ring tracks live input level while listening, so it is obvious
/// whether the microphone is actually hearing anything — the failure the
/// user is least equipped to diagnose.
struct MicOrb: View {

    let state: ConversationController.State
    let level: Float
    let action: () -> Void

    @State private var pulse = false

    private var tint: Color {
        switch state {
        case .listening: return .red
        case .thinking: return .orange
        case .speaking: return .green
        case .failed: return .secondary
        case .cold: return .secondary
        case .idle: return .accentColor
        }
    }

    private var symbol: String {
        switch state {
        case .listening: return "waveform"
        case .thinking: return "ellipsis"
        case .speaking: return "speaker.wave.2.fill"
        case .cold: return "hourglass"
        default: return "mic.fill"
        }
    }

    private var isEnabled: Bool {
        state != .cold && state != .thinking
    }

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .stroke(tint.opacity(0.25), lineWidth: 3)
                    .frame(width: 108, height: 108)

                // Level ring — only meaningful while listening.
                Circle()
                    .stroke(tint.opacity(0.55), lineWidth: 4)
                    .frame(width: 108, height: 108)
                    .scaleEffect(state == .listening ? 1 + CGFloat(level) * 0.32 : 1)
                    .opacity(state == .listening ? 1 : 0)
                    .animation(.easeOut(duration: 0.12), value: level)

                Circle()
                    .fill(tint.opacity(0.16))
                    .frame(width: 88, height: 88)
                    .scaleEffect(pulse && state == .thinking ? 1.08 : 1)

                Image(systemName: symbol)
                    .font(.system(size: 32, weight: .medium))
                    .foregroundStyle(tint)
                    .symbolRenderingMode(.hierarchical)
            }
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .onAppear {
            withAnimation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true)) {
                pulse = true
            }
        }
        .accessibilityLabel(accessibilityLabel)
    }

    private var accessibilityLabel: String {
        switch state {
        case .listening: return "Stop listening"
        case .speaking: return "Interrupt and speak"
        case .thinking: return "Generating a reply"
        case .cold: return "Engine starting"
        default: return "Start listening"
        }
    }
}
