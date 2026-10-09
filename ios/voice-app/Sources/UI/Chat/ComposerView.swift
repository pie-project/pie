import SwiftUI

/// ChatGPT's composer: "+" on the left, "Ask anything" growing to seven
/// lines, then the microphone and one round button that is voice mode,
/// send or stop. While dictating the same row becomes the recording strip:
/// ✕ where "+" was, a live waveform over the field, the clock where the
/// microphone was, and ✓ in the round button.
///
/// Every part keeps its place, so a change of state never moves a
/// neighbour: the round button swaps only its glyph, and the microphone's
/// slot stays reserved while the microphone is hidden, so the first
/// letter typed does not re-wrap the text.
struct ComposerView: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var dictation: DictationController
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The text field's shown height. It follows the field's natural height
    /// through an animation, so a new line grows the composer smoothly
    /// instead of in 22 pt jumps, and the transcript above follows; nil
    /// until measured.
    @State private var fieldHeight: CGFloat?
    /// The pending attachments as shown. The chat changes its list outside
    /// any animation (an import finishes in the background); copying it
    /// here inside one lets the tiles pop in and out and the composer grow
    /// and shrink with them instead of jumping.
    @State private var shownAttachments: [Attachment] = []
    @State private var showsImportTile = false

    /// A new line: nearly instant, never a jump.
    private static let growth = Animation.smooth(duration: 0.18)
    /// Attachment tiles arrive quickly and leave a little slower, so the
    /// neighbours' slide reads.
    private static let attachmentIn = Animation.snappy(duration: 0.25)
    private static let attachmentOut = Animation.smooth(duration: 0.25)

    var body: some View {
        box
            .onAppear {
                shownAttachments = chat.pendingAttachments
                showsImportTile = chat.isImportingAttachment
            }
            .onChange(of: chat.pendingAttachments.map(\.id)) { old, new in
                withMotion(new.count >= old.count ? Self.attachmentIn : Self.attachmentOut) {
                    shownAttachments = chat.pendingAttachments
                }
            }
            .onChange(of: chat.isImportingAttachment) { _, importing in
                withMotion(importing ? Self.attachmentIn : Self.attachmentOut) {
                    showsImportTile = importing
                }
            }
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

    /// The rounded box: attachments waiting to go, then the row.
    private var box: some View {
        VStack(alignment: .leading, spacing: 0) {
            if !shownAttachments.isEmpty || showsImportTile {
                PendingAttachmentsRow(attachments: shownAttachments, showsImportTile: showsImportTile)
                    .transition(.opacity)
            }
            row
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
    }

    // MARK: - The row

    private var row: some View {
        HStack(alignment: .bottom, spacing: 0) {
            leadingButton
            ZStack(alignment: .bottomLeading) {
                field
                    .opacity(showsDictation ? 0 : 1)
                    .allowsHitTesting(!showsDictation)
                    .accessibilityHidden(showsDictation)
                if showsDictation {
                    DictationWaveform(meter: dictation.meter, isPaused: dictation.isPaused || dictation.isFinishing)
                        .padding(.horizontal, 6)
                        .transition(.opacity)
                }
            }
            microphoneSlot
            roundButton
        }
    }

    /// "+", or ✕ while dictating; one pops in where the other was.
    private var leadingButton: some View {
        ZStack {
            if showsDictation {
                ComposerIconButton(symbol: "xmark", style: .plain, label: "Cancel dictation") {
                    dictation.cancel()
                }
                .transition(Self.buttonSwap)
            } else {
                ComposerIconButton(symbol: "plus", style: .outlined, label: "Add photos and files") {
                    KeyboardDismissal.dismiss()
                    router.isAttachmentSheetPresented = true
                }
                .transition(Self.buttonSwap)
            }
        }
    }

    private var field: some View {
        TextField("Ask anything", text: $chat.draft, axis: .vertical)
            .lineLimit(1...7)
            .font(.body)
            .foregroundStyle(Theme.ink)
            .padding(.horizontal, 6)
            .padding(.vertical, 11)
            .frame(minHeight: 44)
            // Its natural height, whatever the frame below allows right now.
            .fixedSize(horizontal: false, vertical: true)
            .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height in
                follow(fieldHeight: height)
            }
            // Pinned to the top while the frame catches up, so on a new line
            // the text glides up with the composer's top edge; the line being
            // added shows as the growth (0.18 s) uncovers it.
            .frame(height: fieldHeight, alignment: .top)
            .clipped()
    }

    /// The microphone, or the clock while dictating, in a slot that keeps
    /// its width either way.
    private var microphoneSlot: some View {
        ZStack {
            if showsDictation {
                Text(Self.clock(dictation.elapsed))
                    .font(.subheadline)
                    .monospacedDigit()
                    .foregroundStyle(Theme.secondaryInk)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .accessibilityLabel("Dictating, \(Int(dictation.elapsed)) seconds")
                    .transition(.opacity)
            } else if showsMicrophone {
                ComposerIconButton(symbol: "mic", style: .plain, label: "Dictate") {
                    startDictation()
                }
                .transition(Self.buttonSwap)
            }
        }
        .frame(width: 44, height: 44)
        // Typing hides the microphone and clearing shows it again; neither
        // happens inside an animation of its own. Local to the slot, whose
        // size never changes, so nothing else moves.
        .animation(Motion.control, value: showsMicrophone)
    }

    /// One circle for four roles. Only its glyph changes (see
    /// `ComposerIconButton`), so voice becomes send becomes stop in place.
    private var roundButton: some View {
        let role = roundButtonRole
        return ComposerIconButton(
            symbol: role.symbol,
            style: .filled,
            label: role.label,
            isEnabled: role != .send || chat.canSend,
            showsProgress: role == .finishDictation && dictation.isFinishing
        ) {
            switch role {
            case .finishDictation:
                Haptics.tap(enabled: settings.haptics)
                dictation.finish(into: chat)
            case .stop:
                Haptics.tap(enabled: settings.haptics)
                chat.stop()
            case .send:
                send()
            case .voice:
                KeyboardDismissal.dismiss()
                router.isVoiceModePresented = true
            }
        }
    }

    // MARK: - State

    private enum RoundButtonRole {
        case finishDictation, stop, send, voice

        var symbol: String {
            switch self {
            case .finishDictation: return "checkmark"
            case .stop: return "stop.fill"
            case .send: return "arrow.up"
            case .voice: return "waveform"
            }
        }

        var label: String {
            switch self {
            case .finishDictation: return "Finish dictation"
            case .stop: return "Stop generating"
            case .send: return "Send"
            case .voice: return "Start voice mode"
            }
        }
    }

    private var roundButtonRole: RoundButtonRole {
        if showsDictation { return .finishDictation }
        if chat.isGenerating { return .stop }
        if hasSomethingToSend { return .send }
        return .voice
    }

    /// The recording strip shows while dictating, except while a refusal
    /// already known when the microphone was tapped is being confirmed:
    /// then the tap only brings up the banner, instead of flashing the
    /// strip in and out. A microphone that fails once listening pauses the
    /// session, and the strip stays for done or cancel.
    private var showsDictation: Bool {
        dictation.isDictating
            && (Self.unavailableReason(dictation.availability) == nil || dictation.isPaused)
    }

    private var showsMicrophone: Bool {
        !chat.isGenerating && !hasSomethingToSend
    }

    /// Text, an attachment, or one on its way (then send shows, disabled,
    /// until it is read, as ChatGPT's does during an upload).
    private var hasSomethingToSend: Bool {
        !chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !chat.pendingAttachments.isEmpty
            || chat.isImportingAttachment
    }

    // MARK: - Actions

    private func send() {
        guard chat.canSend else { return }
        Haptics.tap(enabled: settings.haptics)
        // First, so the composer lands with the keyboard before the send's
        // own animations start (see `KeyboardDismissal`).
        KeyboardDismissal.dismiss()
        // Back to one line at once, as ChatGPT's field does on send: the
        // next measurement finds the draft empty and is taken as it is.
        fieldHeight = nil
        // Not wrapped in an animation here: the chat animates the message
        // going out itself.
        chat.send()
    }

    private func startDictation() {
        Haptics.tap(enabled: settings.haptics)
        KeyboardDismissal.dismiss()
        // The microphone would hear the reply being read and transcribe it.
        chat.stopReadingAloud()
        // A refusal the controller already knows about will not change
        // `availability` again, so say why here rather than waiting for it.
        if let reason = Self.unavailableReason(dictation.availability) {
            chat.banner = reason
        }
        dictation.start()
    }

    /// Follows the field's natural height: animated while typing or when
    /// dictated text arrives; at once on the first measurement, when the
    /// field has been emptied (sent or cleared), and with Reduce Motion.
    private func follow(fieldHeight height: CGFloat) {
        guard height != fieldHeight else { return }
        if fieldHeight == nil || chat.draft.isEmpty || reduceMotion {
            fieldHeight = height
        } else {
            withAnimation(Self.growth) { fieldHeight = height }
        }
    }

    // MARK: - Helpers

    /// A button giving way to another in the same place: the newcomer pops
    /// in from 80 %, the one leaving fades.
    private static var buttonSwap: AnyTransition {
        AnyTransition.popIn(from: 0.8).animation(Motion.control)
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
