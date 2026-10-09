import SwiftUI
import UIKit

/// A reply: no bubble, full width, Markdown. While it is on its way it
/// shows the pulsing dot, then "Thinking" (in thinking mode), then the
/// text as it streams; once finished, the action row and, if enabled, the
/// engine stats.
///
/// The dot sits over the row's top-left corner, where the first word (or
/// "Thinking") will be. It hands over quickly, as ChatGPT's does: it
/// shrinks away in 0.1 s, and what replaces it starts fading in 0.08 s
/// later, so the dot is never drawn over a half-faded first letter (it
/// was, for about 80 ms, when both faded at once). The controls fade in a
/// moment after the last word, below the reply, without moving it.
///
/// The controller changes a generating reply without an animation (see
/// `ChatController`), so every piece here that comes or goes carries its
/// own.
///
/// Equatable on its values (not its closures or binding), so the list can
/// skip every reply but the one that is streaming.
struct AssistantMessageRow: View, Equatable {
    let message: StoredMessage
    /// The controller's phase, for the message being generated; nil for
    /// every other message.
    let livePhase: ChatController.ReplyPhase?
    let isReadingAloud: Bool
    let showsStats: Bool
    let actions: MessageActions
    let sheet: Binding<MessageSheet?>

    static func == (lhs: AssistantMessageRow, rhs: AssistantMessageRow) -> Bool {
        lhs.message == rhs.message
            && lhs.livePhase == rhs.livePhase
            && lhs.isReadingAloud == rhs.isReadingAloud
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
                .transition(Self.replacesDot)
            }

            if !message.text.isEmpty {
                MarkdownView(text: message.text, isStreaming: message.isStreaming)
                    .equatable()
                    .contentShape(Rectangle())
                    .contextMenu { contextMenuItems }
                    .transition(Self.replacesDot)
            }

            if !message.isStreaming {
                footer
                    .transition(.asymmetric(
                        insertion: .opacity.animation(Motion.fadeIn.delay(0.06)),
                        removal: .opacity.animation(Motion.fadeOut)
                    ))
            }
        }
        .frame(maxWidth: .infinity, minHeight: isWaiting ? PulsingDot.lineHeight : nil, alignment: .topLeading)
        .overlay(alignment: .topLeading) {
            if isWaiting {
                PulsingDot()
                    .transition(Self.dotLeaving)
            }
        }
    }

    /// The dot arrives by itself (`PulsingDot`) and leaves shrinking
    /// toward its centre while it fades, gone in 0.1 s. With Reduce Motion
    /// it only fades.
    private static var dotLeaving: AnyTransition {
        .asymmetric(
            insertion: .identity,
            removal: .opacity.combined(with: .scale(scale: Motion.prefersReduced ? 1 : 0.4))
                .animation(.easeOut(duration: 0.1))
        )
    }

    /// The first words (or "Thinking") start fading in just as the dot
    /// has nearly gone. The words then also fade in one by one
    /// (`RevealText`); this short fade of the whole block only delays them.
    private static let replacesDot = AnyTransition.asymmetric(
        insertion: .opacity.animation(.easeOut(duration: 0.12).delay(0.08)),
        removal: .opacity.animation(Motion.fadeOut)
    )

    /// Sent, and nothing has come back yet.
    private var isWaiting: Bool {
        message.isStreaming && message.text.isEmpty && !isThinking
    }

    private var isThinking: Bool {
        if case .thinking = livePhase { return true }
        return false
    }

    /// A stopped reply keeps its words and gets its controls, with no
    /// "Stopped" label, as in ChatGPT now. Only a reply stopped before its
    /// first word says so, since otherwise there would be nothing above
    /// its controls.
    private var footer: some View {
        VStack(alignment: .leading, spacing: 0) {
            if message.wasStopped && message.text.isEmpty {
                Label("Stopped", systemImage: "stop.circle")
                    .font(.footnote)
                    .foregroundStyle(Theme.tertiaryInk)
            }
            MessageActionBar(
                message: message,
                isReadingAloud: isReadingAloud,
                actions: actions
            )
            if showsStats, let stats = message.stats {
                EngineStatsCaption(stats: stats)
            }
        }
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
            RegenerateContextButton(regenerate: { actions.regenerate(message.id, nil) })
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

/// The context menu's Regenerate, disabled while a reply is being
/// generated. It reads that from the environment, so the flag flipping at
/// every send and finish does not redraw every reply in the list.
private struct RegenerateContextButton: View {
    let regenerate: () -> Void
    @Environment(\.canRegenerate) private var canRegenerate

    var body: some View {
        Button(action: regenerate) {
            Label("Regenerate", systemImage: "arrow.clockwise")
        }
        .disabled(!canRegenerate)
    }
}
