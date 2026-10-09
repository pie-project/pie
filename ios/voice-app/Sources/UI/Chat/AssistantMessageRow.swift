import SwiftUI
import UIKit

/// A reply: no bubble, full width, Markdown. While it is on its way it
/// shows the pulsing dot, then "Thinking", then the text with a streaming
/// dot; once finished, the action row and, if enabled, the engine stats.
///
/// Equatable on its values (not its closures or binding), so the list can
/// skip every reply but the one that is streaming.
struct AssistantMessageRow: View, Equatable {
    let message: StoredMessage
    /// The controller's phase, for the message being generated; nil for
    /// every other message.
    let livePhase: ChatController.ReplyPhase?
    let isReadingAloud: Bool
    let canRegenerate: Bool
    let showsStats: Bool
    let actions: MessageActions
    let sheet: Binding<MessageSheet?>

    static func == (lhs: AssistantMessageRow, rhs: AssistantMessageRow) -> Bool {
        lhs.message == rhs.message
            && lhs.livePhase == rhs.livePhase
            && lhs.isReadingAloud == rhs.isReadingAloud
            && lhs.canRegenerate == rhs.canRegenerate
            && lhs.showsStats == rhs.showsStats
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if isThinking || !message.reasoning.isEmpty {
                ReasoningDisclosure(
                    reasoning: message.reasoning,
                    isThinking: isThinking,
                    thoughtSeconds: message.thoughtSeconds
                )
            }

            if message.isStreaming && message.text.isEmpty {
                if !isThinking { PulsingDot() }
            } else if !message.text.isEmpty {
                MarkdownView(text: message.text, isStreaming: message.isStreaming)
                    .equatable()
                    .contentShape(Rectangle())
                    .contextMenu { contextMenuItems }
            }

            if message.wasStopped {
                Label("Stopped", systemImage: "stop.circle")
                    .font(.footnote)
                    .foregroundStyle(Theme.tertiaryInk)
            }

            if !message.isStreaming {
                VStack(alignment: .leading, spacing: 0) {
                    MessageActionBar(
                        message: message,
                        isReadingAloud: isReadingAloud,
                        canRegenerate: canRegenerate,
                        actions: actions
                    )
                    if showsStats, let stats = message.stats {
                        EngineStatsCaption(stats: stats)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var isThinking: Bool {
        if case .thinking = livePhase { return true }
        return false
    }

    @ViewBuilder
    private var contextMenuItems: some View {
        if !message.isStreaming {
            Button {
                UIPasteboard.general.string = message.text
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
            Button {
                sheet.wrappedValue = .selectText(message.text)
            } label: {
                Label("Select Text", systemImage: "selection.pin.in.out")
            }
            Button {
                actions.toggleReadAloud(message.id)
            } label: {
                Label(isReadingAloud ? "Stop Reading" : "Read Aloud", systemImage: isReadingAloud ? "stop.fill" : "speaker.wave.2")
            }
            Button {
                actions.regenerate(message.id, nil)
            } label: {
                Label("Regenerate", systemImage: "arrow.clockwise")
            }
            .disabled(!canRegenerate)
            Button {
                actions.setFeedback(message.feedback == .good ? nil : .good, message.id)
            } label: {
                Label("Good response", systemImage: message.feedback == .good ? "hand.thumbsup.fill" : "hand.thumbsup")
            }
            Button {
                actions.setFeedback(message.feedback == .bad ? nil : .bad, message.id)
            } label: {
                Label("Bad response", systemImage: message.feedback == .bad ? "hand.thumbsdown.fill" : "hand.thumbsdown")
            }
        }
    }
}
