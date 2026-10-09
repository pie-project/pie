import SwiftUI
import UIKit

/// Full-screen voice mode, laid out like ChatGPT's: the orb in the middle,
/// live captions under it, and mute, more and end along the bottom, in the
/// palette of Lin Zhong's site.
///
/// Talking while Pie speaks interrupts it (the microphone stays open
/// behind the echo canceller); so does tapping the orb.
///
/// Motion, after ChatGPT's advanced voice mode:
/// - entering, the orb grows in from 60% as it fades in, and the controls
///   follow a beat later, fading in as they rise; a soft haptic marks the
///   moment the microphone is listening;
/// - leaving, the orb shrinks a little and fades and the controls fade,
///   a beat quicker than the screen itself fades back to the chat;
/// - in between, most changes come from the audio side (a reply starting,
///   the user talking over it), not from a tap, so each piece animates its
///   own: the orb eases between moods frame by frame (`VoiceOrb`), the
///   status line fades one message out before the next fades in, the mute
///   button swaps its glyph and fill, and new caption words fade in
///   (`VoiceCaptionsView`).
///
/// The screen itself fades in and out over the chat (0.35 s, as ChatGPT's
/// separate voice mode does); the orb's and the controls' own fades run
/// inside that one, so the orb arrives a moment after the background and
/// leaves a moment before it.
///
/// The orb, the status line and the captions each sit in a slot of fixed
/// size, so nothing that changes in one of them moves the others.
struct VoiceModeView: View {

    @EnvironmentObject private var voice: VoiceModeController
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var dictation: DictationController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// "Voice settings" closes voice mode first; Settings is presented once
    /// voice mode has gone, so the sheet does not rise while it is still
    /// fading out.
    @State private var opensSettingsAfterClose = false
    /// The orb has grown in; false before voice mode appears and once it
    /// is closing.
    @State private var orbShown = false
    /// The header, captions and controls have faded in.
    @State private var controlsShown = false
    /// `close()` has run: the orb is leaving rather than arriving, so it
    /// shrinks less than it grew.
    @State private var isClosing = false

    /// Entering: the orb grows from 60% while fading in (ChatGPT, estimated).
    private static let orbEntrance = Animation.smooth(duration: 0.5)
    /// The controls follow a beat after the orb, rising 20 points.
    private static let controlsEntrance = Animation.smooth(duration: 0.3).delay(0.1)
    /// Leaving: the orb (to 85%) and the controls fade out in 0.25 s, so
    /// they are gone a beat before the screen's own 0.35 s fade
    /// (`AppRouter.voiceModeCrossfade`) ends.
    private static let exit = Animation.smooth(duration: 0.25)

    var body: some View {
        GeometryReader { geometry in
            // About 60% of the width, as ChatGPT's orb, unless the screen
            // is short.
            let diameter = min(geometry.size.width * 0.6, geometry.size.height * 0.34)
            VStack(spacing: 0) {
                header
                    .padding(.horizontal, 20)
                    .padding(.top, 8)
                    .opacity(controlsShown ? 1 : 0)

                Spacer(minLength: 16)
                stage(diameter: diameter)
                Spacer(minLength: 16)

                if settings.voiceCaptions {
                    VoiceCaptionsView(captions: voice.captions)
                        .frame(height: geometry.size.height * 0.24)
                        .opacity(controlsShown ? 1 : 0)
                        .transition(.opacity)
                }

                controls
                    .padding(.horizontal, 32)
                    .padding(.top, 16)
                    .padding(.bottom, 8)
                    .opacity(controlsShown ? 1 : 0)
                    .offset(y: controlsShown || reduceMotion ? 0 : 20)
            }
        }
        .background(Theme.background.ignoresSafeArea())
        .onAppear {
            // A conversation can outlast the auto-lock delay without a
            // single touch.
            UIApplication.shared.isIdleTimerDisabled = true
            VoiceHaptics.prepare(enabled: settings.haptics)
            // One transaction with the orb's entrance, for the same reason
            // as in `close()`: the conversation's published changes (the
            // status, the mood) land in the same moment as the screen's
            // fade-in, and must not cancel it.
            withMotion(Self.orbEntrance) {
                // The sidebar can open voice mode while the composer is
                // still dictating; both share the one microphone, and the
                // composer would otherwise come back showing a dictation
                // with no session.
                dictation.cancel()
                voice.start()
                orbShown = true
            }
            withMotion(Self.controlsEntrance) { controlsShown = true }
        }
        .onDisappear {
            voice.end()
            UIApplication.shared.isIdleTimerDisabled = false
            if opensSettingsAfterClose {
                opensSettingsAfterClose = false
                DispatchQueue.main.async { router.isSettingsPresented = true }
            }
        }
        .onChange(of: voice.isConnecting) { wasConnecting, isConnecting in
            // The microphone is open and listening: ChatGPT marks the
            // moment its voice session connects with a soft tap.
            guard wasConnecting, !isConnecting, voice.phase == .listening, !voice.isMuted else { return }
            VoiceHaptics.connected(enabled: settings.haptics)
        }
        .onChange(of: isFailed) { _, failed in
            // The notice explains a missing permission on its own.
            if failed, inputNotice == nil { Haptics.failure(enabled: settings.haptics) }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Pie")
                    .font(Theme.serif(22))
                    .foregroundStyle(Theme.ink)
                Text("On this iPhone · \(PieRuntimeConfig.modelDescription)")
                    .font(.footnote)
                    .foregroundStyle(Theme.secondaryInk)
            }
            Spacer()
            Button {
                Haptics.selection(enabled: settings.haptics)
                setCaptions(!settings.voiceCaptions)
            } label: {
                Image(systemName: settings.voiceCaptions ? "captions.bubble.fill" : "captions.bubble")
                    .font(.system(size: 18, weight: .medium))
                    .contentTransition(.symbolEffect(.replace))
                    .foregroundStyle(settings.voiceCaptions ? Theme.accent : Theme.secondaryInk)
                    .frame(width: 44, height: 44)
                    .background(Circle().fill(settings.voiceCaptions ? Theme.accentWash : Theme.surface))
                    .animation(Motion.crossfade, value: settings.voiceCaptions)
            }
            .buttonStyle(PressScaleButtonStyle())
            .accessibilityLabel("Captions")
            .accessibilityValue(settings.voiceCaptions ? "On" : "Off")
        }
    }

    // MARK: - Orb and status

    /// The orb and, under it, the status line, or the notice when the
    /// microphone cannot be used. The notice shrinks the orb in place
    /// rather than swapping in a second one.
    private func stage(diameter: CGFloat) -> some View {
        let notice = inputNotice
        return VStack(spacing: 28) {
            orb(diameter: diameter, size: notice == nil ? 1 : 0.62)
            Group {
                if let notice {
                    unavailableNotice(notice)
                        .transition(.fadeRise(8))
                } else {
                    VoiceStatusLine()
                        // Room for three lines, so a long failure message
                        // never pushes the orb up.
                        .frame(maxWidth: .infinity)
                        .frame(height: 66, alignment: .top)
                        .padding(.horizontal, 24)
                        .transition(.opacity)
                }
            }
            .opacity(orbShown ? 1 : 0)
        }
        // The notice arrives from the permission check, not from a tap, so
        // its change animates here.
        .animation(Motion.reduced(Motion.content, reduceMotion), value: notice == nil)
    }

    /// `size` shrinks the orb in place (for the notice) without changing
    /// the diameter it draws at, so its clouds and rim keep their scale.
    private func orb(diameter: CGFloat, size: CGFloat) -> some View {
        let entrance: CGFloat = orbShown || reduceMotion ? 1 : (isClosing ? 0.85 : 0.6)
        return Button {
            Haptics.tap(enabled: settings.haptics)
            voice.tapOrb()
        } label: {
            VoiceOrb(mood: mood, levels: voice.levels, diameter: diameter)
                .contentShape(Circle())
        }
        .buttonStyle(PressScaleButtonStyle())
        .accessibilityLabel(orbAccessibilityLabel)
        .allowsHitTesting(size == 1)
        .accessibilityHidden(size != 1)
        .scaleEffect(size * entrance)
        .opacity(orbShown ? 1 : 0)
        .frame(width: diameter, height: diameter * size)
    }

    private var mood: VoiceOrb.Mood {
        if inputNotice != nil { return .muted }
        // Before `start()` has run and once `end()` has, the orb is on its
        // way in or out; dim and small suits both.
        if !voice.isActive || voice.isConnecting { return .connecting }
        switch voice.phase {
        case .failed: return .failed
        case .thinking: return .thinking
        case .speaking: return .speaking
        case .listening, .idle: return voice.isMuted ? .muted : .listening
        }
    }

    private var orbAccessibilityLabel: String {
        switch voice.phase {
        case .speaking, .thinking: return "Interrupt"
        case .listening: return "Done talking"
        case .idle, .failed: return "Start listening"
        }
    }

    private var isFailed: Bool {
        if case .failed = voice.phase { return true }
        return false
    }

    // MARK: - Speech input unavailable

    private struct InputNotice {
        let reason: String
        /// Permission was refused, so the system Settings app can fix it.
        let isDenied: Bool
    }

    /// Why the microphone cannot be used, while nothing else (a sample
    /// question, say) is driving the orb.
    private var inputNotice: InputNotice? {
        switch voice.phase {
        case .idle, .failed: break
        case .listening, .thinking, .speaking: return nil
        }
        switch voice.availability {
        case .denied(let reason)?: return InputNotice(reason: reason, isDenied: true)
        case .unavailable(let reason)?: return InputNotice(reason: reason, isDenied: false)
        case .ready?, nil: return nil
        }
    }

    private func unavailableNotice(_ notice: InputNotice) -> some View {
        VStack(spacing: 18) {
            Text(notice.reason)
                .font(.callout)
                .foregroundStyle(Theme.secondaryInk)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            if notice.isDenied, let url = URL(string: UIApplication.openSettingsURLString) {
                Link("Open Settings", destination: url)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(Theme.accent)
            }
            Button {
                Haptics.tap(enabled: settings.haptics)
                close()
            } label: {
                Text("Type instead")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(Theme.onAccent)
                    .padding(.horizontal, 28)
                    .padding(.vertical, 12)
                    .background(Capsule().fill(Theme.accentFill))
            }
            .buttonStyle(PressScaleButtonStyle())
        }
    }

    // MARK: - Controls

    private var controls: some View {
        HStack {
            Button {
                Haptics.tap(enabled: settings.haptics)
                voice.toggleMute()
            } label: {
                Image(systemName: voice.isMuted ? "mic.slash.fill" : "mic.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .contentTransition(.symbolEffect(.replace))
                    .foregroundStyle(voice.isMuted ? Theme.onAccent : Theme.ink)
                    .frame(width: 64, height: 64)
                    .background(Circle().fill(voice.isMuted ? Theme.red : Theme.surface))
                    // Here rather than at the tap: muting also changes
                    // when voice mode ends or the orb is tapped.
                    .animation(Motion.crossfade, value: voice.isMuted)
            }
            .buttonStyle(PressScaleButtonStyle())
            .accessibilityLabel(voice.isMuted ? "Unmute microphone" : "Mute microphone")

            Spacer()

            moreMenu

            Spacer()

            Button {
                Haptics.tap(enabled: settings.haptics)
                close()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(Theme.onAccent)
                    .frame(width: 64, height: 64)
                    .background(Circle().fill(Theme.red))
            }
            .buttonStyle(PressScaleButtonStyle())
            .accessibilityLabel("End voice mode")
        }
    }

    /// A system menu, so it keeps the system's own morph and feedback.
    private var moreMenu: some View {
        Menu {
            if voice.hasSampleQuestions {
                Button {
                    Haptics.tap(enabled: settings.haptics)
                    voice.askSampleQuestion()
                } label: {
                    Label("Ask a sample question", systemImage: "waveform")
                }
            }
            Toggle(isOn: Binding(get: { settings.voiceCaptions }, set: setCaptions)) {
                Label("Captions", systemImage: "captions.bubble")
            }
            Button {
                opensSettingsAfterClose = true
                close()
            } label: {
                Label("Voice settings", systemImage: "gearshape")
            }
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(Theme.ink)
                .frame(width: 64, height: 64)
                .background(Circle().fill(Theme.surface))
        }
        .menuOrder(.fixed)
        .accessibilityLabel("More")
    }

    // MARK: - Actions

    /// Captions on or off, from the header button or the menu: the
    /// captions fade and the orb glides to its new place.
    private func setCaptions(_ isOn: Bool) {
        withMotion(Motion.content) { settings.voiceCaptions = isOn }
    }

    /// The conversation stops at once; the orb shrinks and fades and the
    /// controls fade while the screen fades back to the chat.
    ///
    /// All of it is one transaction, `voice.end()` included: its published
    /// changes (the status emptying, the captions clearing, the mute
    /// button resetting) are made in the same moment as the screen's fade,
    /// and an unanimated published change in the same run-loop turn as an
    /// animated one can cancel that animation. The orb and the controls
    /// get their own quicker ease, nested inside.
    private func close() {
        withMotion(AppRouter.voiceModeCrossfade) {
            voice.end()
            withMotion(Self.exit) {
                isClosing = true
                orbShown = false
                controlsShown = false
            }
            router.isVoiceModePresented = false
        }
    }
}

/// The line under the orb: what voice mode is doing. A view of its own
/// because it watches the chat controller for the model's loading
/// progress, and that controller publishes every streamed token: only
/// this line redraws for them, not the whole screen.
///
/// A new message replaces the old one in two overlapping steps rather than
/// one crossfade: the old fades out in 0.14 s and the new fades in once
/// the old is mostly gone. Crossfaded in place, two different strings
/// showed at half strength on top of each other for a moment ("Speaking -
/// talk or tap to interrupt" through "Listening"), their letters
/// interleaved.
private struct VoiceStatusLine: View {
    @EnvironmentObject private var voice: VoiceModeController
    @EnvironmentObject private var chat: ChatController

    /// How long the new message waits for the old one to fade: by then
    /// the old one is all but gone (at 0.1 s the two still showed faintly
    /// together for two frames, recorded).
    private static let handOver: TimeInterval = 0.12

    var body: some View {
        let text = text
        // Top-aligned, so a one-line message and a three-line one start at
        // the same place while they overlap.
        ZStack(alignment: .top) {
            Text(text)
                .font(.callout.weight(.medium))
                .foregroundStyle(isFailed ? Theme.destructive : Theme.secondaryInk)
                .multilineTextAlignment(.center)
                .lineLimit(3)
                // The loading line keeps its view while its seconds count
                // up, so only the digits roll instead of the whole line
                // blinking out and in every second.
                .contentTransition(.numericText())
                // A new view for each message, so the old one can fade out
                // while the new one fades in after it.
                .id(isLoading ? "loading" : text)
                .transition(
                    .asymmetric(
                        insertion: .opacity.animation(Motion.fadeIn.delay(Self.handOver)),
                        removal: .opacity.animation(Motion.fadeOut)
                    )
                )
        }
        .animation(Motion.crossfade, value: text)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.updatesFrequently)
    }

    private var text: String {
        // Before `start()` and after `end()`: nothing, so the line simply
        // fades as voice mode opens and closes.
        guard voice.isActive else { return "" }
        switch voice.phase {
        case .failed(let message):
            return message
        case .speaking:
            return voice.isMuted ? "Speaking - tap to interrupt" : "Speaking - talk or tap to interrupt"
        case .thinking:
            return "Thinking…"
        case .listening, .idle:
            if voice.isMuted { return "Mic is off" }
            // A question asked before the model is up cannot be answered.
            if case .booting(let seconds) = chat.engineState {
                return "Loading \(PieRuntimeConfig.modelDescription)… \(seconds) s"
            }
            return voice.isConnecting ? "Starting…" : "Listening"
        }
    }

    private var isFailed: Bool {
        if case .failed = voice.phase { return true }
        return false
    }

    /// The line is counting the model's loading time.
    private var isLoading: Bool {
        guard case .booting = chat.engineState else { return false }
        return text.hasPrefix("Loading")
    }
}

/// The soft tap when voice mode starts listening, like ChatGPT's when its
/// voice session connects. Prepared on entry, so it is on time.
@MainActor
private enum VoiceHaptics {
    private static let soft = UIImpactFeedbackGenerator(style: .soft)

    static func prepare(enabled: Bool) {
        guard enabled else { return }
        soft.prepare()
    }

    static func connected(enabled: Bool) {
        guard enabled else { return }
        soft.impactOccurred(intensity: 0.9)
    }
}
