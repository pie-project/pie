import SwiftUI

/// ChatGPT's composer: "+" on the left, "Ask anything" growing to seven
/// lines, then the microphone and one round button that is voice mode,
/// send or stop. While dictating the same row becomes the recording strip:
/// ✕ where "+" was, a live waveform over the field, the clock where the
/// microphone was, and ✓ in the round button.
///
/// Every part keeps its place, so a change of state never moves a
/// neighbour: the round button swaps only its glyph, and the microphone
/// stays while the user types, as ChatGPT's does, so the first letter
/// typed changes nothing but that glyph. (It hides only while a reply is
/// generating, as before; its slot stays reserved.)
///
/// Where one piece gives way to another in the same place (the field and
/// the recording strip, the microphone and the clock), the one leaving
/// fades in 0.14 s and the one arriving starts 0.07 s later, so the two
/// are never both legible: overlapping fades drew "+" and "✕" as an
/// asterisk and the dotted strip through "Ask anything" (recorded).
struct ComposerView: View {
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var dictation: DictationController
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The text area's shown height. It follows the text's natural height
    /// through an animation, so a new line grows the composer smoothly
    /// instead of in 22 pt jumps, and the transcript above follows; nil
    /// until measured, and set back to nil on send.
    @State private var fieldHeight: CGFloat?
    /// Set by send, whose emptied field returns to one line at once, as
    /// ChatGPT's does. Emptying the field by hand (select all, delete)
    /// collapses it with the same animation as growing. Cleared by the
    /// next measurement of a non-empty draft.
    @State private var collapsesAtOnce = false
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
                // The field stays in the hierarchy (it holds the draft and
                // the keyboard focus) and only fades: out quickly as the
                // strip opens, back in just after the strip has gone.
                field
                    .opacity(showsDictation ? 0 : 1)
                    .animation(showsDictation ? Motion.fadeOut : Self.handOffIn, value: showsDictation)
                    .allowsHitTesting(!showsDictation)
                    .accessibilityHidden(showsDictation)
                if showsDictation {
                    DictationWaveform(meter: dictation.meter, isPaused: dictation.isPaused || dictation.isFinishing)
                        .padding(.horizontal, 6)
                        .transition(Self.handOff(scale: 1))
                }
            }
            microphoneSlot
            roundButton
        }
    }

    /// "+", which turns a quarter of the way round into ✕ while dictating,
    /// its ring fading away. One glyph that turns, so there is never a
    /// moment with both on screen; one button, whose action and label
    /// follow the state. Reduce Motion swaps the glyphs by a fade instead.
    private var leadingButton: some View {
        let cancels = showsDictation
        return Button {
            if cancels {
                dictation.cancel()
            } else {
                KeyboardDismissal.dismiss()
                router.isAttachmentSheetPresented = true
            }
        } label: {
            ZStack {
                Circle()
                    .strokeBorder(Theme.hairline, lineWidth: 1)
                    .opacity(cancels ? 0 : 1)
                Image(systemName: reduceMotion && cancels ? "xmark" : "plus")
                    .font(.system(size: 18, weight: .regular))
                    .foregroundStyle(Theme.ink)
                    .rotationEffect(.degrees(cancels && !reduceMotion ? 45 : 0))
                    .contentTransition(.opacity)
            }
            .frame(width: 36, height: 36)
            .frame(width: 44, height: 44)
            .contentShape(Circle())
            .animation(Motion.control, value: cancels)
        }
        .buttonStyle(PressDimButtonStyle())
        .accessibilityLabel(cancels ? "Cancel dictation" : "Add photos and files")
    }

    /// The text field, clipped to its animated height.
    ///
    /// UIKit draws a wrapped line the moment it wraps, but SwiftUI learns
    /// the taller height a frame or two later, so the growth starts late.
    /// The clip is therefore the text itself (with 1 pt to spare for the
    /// caret), not the padded field: in those frames a new line stays hidden instead of showing
    /// cut off in the bottom padding (recorded), and the growth uncovers
    /// it. Pinned to the top, so the lines above glide up with the
    /// composer's top edge.
    private var field: some View {
        TextField("Ask anything", text: $chat.draft, axis: .vertical)
            .lineLimit(1...7)
            .font(.body)
            .foregroundStyle(Theme.ink)
            .padding(.horizontal, 6)
            .padding(.vertical, Self.clipAllowance)
            // Its natural height, whatever the frame below allows right now.
            .fixedSize(horizontal: false, vertical: true)
            .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height in
                follow(fieldHeight: height)
            }
            .frame(height: fieldHeight, alignment: .top)
            .clipped()
            .padding(.vertical, 11 - Self.clipAllowance)
            .frame(minHeight: 44)
    }

    /// The microphone, or the clock while dictating, in a slot that keeps
    /// its width either way. One leaves before the other arrives
    /// (`handOff`), so the microphone never sits on the clock's digits.
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
                    .transition(Self.handOff(scale: 1))
            } else if showsMicrophone {
                ComposerIconButton(symbol: "mic", style: .plain, label: "Dictate") {
                    startDictation()
                }
                .transition(Self.handOff(scale: reduceMotion ? 1 : 0.8))
            }
        }
        .frame(width: 44, height: 44)
        // A reply starting or ending hides or shows the microphone outside
        // any animation of its own. Local to the slot, whose size never
        // changes, so nothing else moves.
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

    /// Shown while typing, as in ChatGPT, where only the round button
    /// changes as the first letter is typed. Hidden while a reply is
    /// generating, as it always has been here, so dictation (the speech
    /// recogniser) never runs alongside the model.
    private var showsMicrophone: Bool {
        !chat.isGenerating
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
        // Back to one line at once, as ChatGPT's field does on send: with
        // no height set, the field takes its one-line height in the same
        // update that empties it, and the measurements that follow are
        // taken as they are.
        fieldHeight = nil
        collapsesAtOnce = true
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

    /// Follows the text's natural height: animated while typing, deleting
    /// (emptying the field by hand included) and when dictated text
    /// arrives; at once on the first measurement, after a send, and with
    /// Reduce Motion.
    private func follow(fieldHeight height: CGFloat) {
        guard height != fieldHeight else { return }
        if !chat.draft.isEmpty { collapsesAtOnce = false }
        if fieldHeight == nil || collapsesAtOnce || reduceMotion {
            fieldHeight = height
        } else {
            withAnimation(Self.growth) { fieldHeight = height }
        }
    }

    // MARK: - Helpers

    /// Room around the text inside the clip, for the caret, which stands a
    /// little taller than its line. Kept to 1 pt: with 3 pt the caret of a
    /// line not yet given its height peeked out as a tick (recorded).
    private static let clipAllowance: CGFloat = 1

    /// The one arriving in a hand-off: a fade-in that waits until the one
    /// leaving (`Motion.fadeOut`, 0.14 s, mostly gone by half way) has
    /// faded most of the way.
    private static let handOffIn = Motion.fadeIn.delay(0.07)

    /// One piece giving way to another in the same place (strip and
    /// field, clock and microphone). Out: a 0.14 s fade. In: `handOffIn`,
    /// from `scale` of its size. The animations ride on the transition, so
    /// they hold whatever animation the change came in (dictation opens
    /// inside a crossfade; a reply ending, inside none).
    private static func handOff(scale: CGFloat) -> AnyTransition {
        .asymmetric(
            insertion: .opacity.combined(with: .scale(scale: scale)).animation(handOffIn),
            removal: .opacity.animation(Motion.fadeOut)
        )
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
