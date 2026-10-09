import SwiftUI

/// ChatGPT's composer: "+" on the left, "Ask anything" growing to six
/// lines, and on the right dictation and voice mode, the send arrow once
/// there is something to send, or stop while a reply is generating. While
/// dictating, the text field gives way to a live waveform and a timer.
struct ComposerView: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var dictation: DictationController
    @EnvironmentObject private var settings: AppSettings

    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if !chat.pendingAttachments.isEmpty || chat.isImportingAttachment {
                PendingAttachmentsRow()
            }
            if dictation.isDictating {
                dictationRow
            } else {
                inputRow
            }
        }
        .padding(4)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .strokeBorder(Theme.hairline, lineWidth: 0.5)
        }
        .padding(.horizontal, 12)
        .padding(.top, 4)
        .padding(.bottom, 8)
        .onChange(of: dictation.availability) { _, availability in
            if let reason = Self.unavailableReason(availability) { chat.banner = reason }
        }
        .onChange(of: chat.readingAloudMessageID) { _, message in
            // Dictation listens on the plain microphone, with no echo
            // cancellation, and read-aloud plays through the speaker: left
            // running, the recogniser would type the reply into the draft.
            // Finishing keeps what the user had already said.
            if message != nil, dictation.isDictating {
                dictation.finish(into: chat)
            }
        }
    }

    private var inputRow: some View {
        HStack(alignment: .bottom, spacing: 0) {
            ComposerIconButton(symbol: "plus", style: .outlined, label: "Add photos and files") {
                isFocused = false
                router.isAttachmentSheetPresented = true
            }
            TextField("Ask anything", text: $chat.draft, axis: .vertical)
                .lineLimit(1...6)
                .font(.body)
                .foregroundStyle(Theme.ink)
                .focused($isFocused)
                .padding(.horizontal, 6)
                .padding(.vertical, 11)
                .frame(minHeight: 44)
            trailingButtons
        }
    }

    @ViewBuilder
    private var trailingButtons: some View {
        if chat.isGenerating {
            ComposerIconButton(symbol: "stop.fill", style: .filled, label: "Stop generating") {
                chat.stop()
            }
        } else if hasSomethingToSend {
            ComposerIconButton(symbol: "arrow.up", style: .filled, label: "Send", isEnabled: chat.canSend) {
                send()
            }
        } else {
            ComposerIconButton(symbol: "mic", style: .plain, label: "Dictate") {
                startDictation()
            }
            ComposerIconButton(symbol: "waveform", style: .filled, label: "Start voice mode") {
                isFocused = false
                router.isVoiceModePresented = true
            }
        }
    }

    private var dictationRow: some View {
        HStack(spacing: 0) {
            ComposerIconButton(symbol: "xmark", style: .plain, label: "Cancel dictation") {
                dictation.cancel()
            }
            DictationWaveform(level: dictation.level)
                .padding(.horizontal, 6)
            Text(Self.clock(dictation.elapsed))
                .font(.subheadline)
                .monospacedDigit()
                .foregroundStyle(Theme.secondaryInk)
                .padding(.trailing, 4)
                .accessibilityLabel("Dictating, \(Int(dictation.elapsed)) seconds")
            ComposerIconButton(symbol: "checkmark", style: .filled, label: "Finish dictation") {
                dictation.finish(into: chat)
            }
        }
    }

    private var hasSomethingToSend: Bool {
        !chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !chat.pendingAttachments.isEmpty
    }

    private func send() {
        guard chat.canSend else { return }
        ChatHaptics.messageSent(enabled: settings.haptics)
        isFocused = false
        chat.send()
    }

    private func startDictation() {
        isFocused = false
        // The microphone would hear the reply being read and transcribe it.
        chat.stopReadingAloud()
        // A refusal the controller already knows about will not change
        // `availability` again, so say why here rather than waiting for it.
        if let reason = Self.unavailableReason(dictation.availability) {
            chat.banner = reason
        }
        dictation.start()
    }

    private static func unavailableReason(_ availability: VoiceInputAvailability?) -> String? {
        switch availability {
        case .denied(let reason)?, .unavailable(let reason)?:
            return reason
        default:
            return nil
        }
    }

    private static func clock(_ seconds: TimeInterval) -> String {
        let whole = max(0, Int(seconds))
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }
}
