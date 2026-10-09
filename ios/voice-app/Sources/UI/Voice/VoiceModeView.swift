import SwiftUI
import UIKit

/// Full-screen voice mode, laid out like ChatGPT's: the orb in the middle,
/// live captions under it, and mute, more and end along the bottom, in the
/// palette of Lin Zhong's site.
///
/// Talking while Pie speaks interrupts it (the microphone stays open
/// behind the echo canceller); so does tapping the orb.
struct VoiceModeView: View {

    @EnvironmentObject private var voice: VoiceModeController
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var dictation: DictationController

    /// "Voice settings" closes voice mode first; Settings is presented once
    /// the cover has gone, since a sheet cannot be presented while the
    /// cover is still being dismissed.
    @State private var opensSettingsAfterClose = false
    @State private var captionsContentHeight: CGFloat = 0

    private let captionsEnd = "captions-end"

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                header
                    .padding(.horizontal, 20)
                    .padding(.top, 8)

                Spacer(minLength: 16)

                if let notice = inputNotice {
                    unavailableNotice(notice)
                } else {
                    orb(diameter: min(240, geometry.size.height * 0.32))
                    statusLine
                        .padding(.top, 32)
                        .padding(.horizontal, 24)
                }

                Spacer(minLength: 16)

                if settings.voiceCaptions && hasCaptions {
                    captions(maxHeight: geometry.size.height * 0.35)
                        .transition(.opacity)
                }

                controls
                    .padding(.horizontal, 32)
                    .padding(.top, 16)
                    .padding(.bottom, 8)
            }
            .animation(.easeInOut(duration: 0.25), value: settings.voiceCaptions)
            .animation(.easeInOut(duration: 0.25), value: hasCaptions)
        }
        .background(Theme.background.ignoresSafeArea())
        .onAppear {
            // A conversation can outlast the auto-lock delay without a
            // single touch.
            UIApplication.shared.isIdleTimerDisabled = true
            // The sidebar can open voice mode while the composer is still
            // dictating; both share the one microphone, and the composer
            // would otherwise come back showing a dictation with no session.
            dictation.cancel()
            voice.start()
        }
        .onDisappear {
            voice.end()
            UIApplication.shared.isIdleTimerDisabled = false
            if opensSettingsAfterClose {
                opensSettingsAfterClose = false
                DispatchQueue.main.async { router.isSettingsPresented = true }
            }
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
                settings.voiceCaptions.toggle()
            } label: {
                Image(systemName: settings.voiceCaptions ? "captions.bubble.fill" : "captions.bubble")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(settings.voiceCaptions ? Theme.accent : Theme.secondaryInk)
                    .frame(width: 44, height: 44)
                    .background(Circle().fill(settings.voiceCaptions ? Theme.accentWash : Theme.surface))
            }
            .buttonStyle(PressScaleButtonStyle())
            .accessibilityLabel("Captions")
            .accessibilityValue(settings.voiceCaptions ? "On" : "Off")
        }
    }

    // MARK: - Orb and status

    private func orb(diameter: CGFloat) -> some View {
        Button {
            feedback()
            voice.tapOrb()
        } label: {
            VoiceOrb(
                mood: mood,
                inputLevel: voice.inputLevel,
                outputLevel: voice.outputLevel,
                diameter: diameter
            )
            .contentShape(Circle())
        }
        .buttonStyle(PressScaleButtonStyle())
        .accessibilityLabel(orbAccessibilityLabel)
    }

    private var statusLine: some View {
        Text(statusText)
            .font(.callout.weight(.medium))
            .foregroundStyle(isFailed ? Theme.destructive : Theme.secondaryInk)
            .multilineTextAlignment(.center)
            .lineLimit(3)
            .contentTransition(.opacity)
            .animation(.easeInOut(duration: 0.2), value: statusText)
            .accessibilityAddTraits(.updatesFrequently)
    }

    private var mood: VoiceOrb.Mood {
        switch voice.phase {
        case .failed: return .failed
        case .thinking: return .thinking
        case .speaking: return .speaking
        case .listening: return voice.isMuted ? .muted : .listening
        case .idle: return voice.isMuted ? .muted : .idle
        }
    }

    private var statusText: String {
        switch voice.phase {
        case .failed(let message):
            return message
        case .speaking:
            return voice.isMuted ? "Speaking - tap to interrupt" : "Speaking - talk or tap to interrupt"
        case .thinking:
            return "Thinking…"
        case .listening:
            return voice.isMuted ? "Mic is off" : "Listening"
        case .idle:
            if case .booting(let seconds) = chat.engineState {
                return "Loading \(PieRuntimeConfig.modelDescription)… \(seconds) s"
            }
            if voice.isMuted { return "Mic is off" }
            return voice.isActive ? "Starting…" : "Tap to talk"
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
        VStack(spacing: 20) {
            VoiceOrb(mood: .muted, diameter: 150)
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
            Button(action: close) {
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

    // MARK: - Captions

    private var hasCaptions: Bool {
        !voice.userCaption.isEmpty || !voice.assistantCaption.isEmpty
    }

    private func captions(maxHeight: CGFloat) -> some View {
        let overflows = captionsContentHeight > maxHeight
        return ScrollViewReader { reader in
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if !voice.userCaption.isEmpty {
                        Text(voice.userCaption)
                            .foregroundStyle(Theme.secondaryInk)
                    }
                    if !voice.assistantCaption.isEmpty {
                        Text(voice.assistantCaption)
                            .foregroundStyle(Theme.ink)
                    }
                    Color.clear
                        .frame(height: 1)
                        .id(captionsEnd)
                }
                .font(.title3)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 24)
                .padding(.top, 8)
                .background(
                    GeometryReader { proxy in
                        Color.clear.preference(key: CaptionsHeightKey.self, value: proxy.size.height)
                    }
                )
            }
            .scrollIndicators(.hidden)
            .frame(height: min(captionsContentHeight, maxHeight))
            // Older lines fade out under the top edge once the captions
            // are taller than their space.
            .mask(
                LinearGradient(
                    stops: [
                        .init(color: overflows ? .clear : .black, location: 0),
                        .init(color: .black, location: overflows ? 0.18 : 0),
                        .init(color: .black, location: 1),
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .onPreferenceChange(CaptionsHeightKey.self) { height in
                captionsContentHeight = height
            }
            .onChange(of: voice.assistantCaption) { _, _ in
                reader.scrollTo(captionsEnd, anchor: .bottom)
            }
            .onChange(of: voice.userCaption) { _, _ in
                reader.scrollTo(captionsEnd, anchor: .bottom)
            }
            .onChange(of: captionsContentHeight) { _, _ in
                reader.scrollTo(captionsEnd, anchor: .bottom)
            }
        }
    }

    // MARK: - Controls

    private var controls: some View {
        HStack {
            Button {
                feedback()
                voice.toggleMute()
            } label: {
                Image(systemName: voice.isMuted ? "mic.slash.fill" : "mic.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(voice.isMuted ? Theme.onAccent : Theme.ink)
                    .frame(width: 64, height: 64)
                    .background(Circle().fill(voice.isMuted ? Theme.red : Theme.surface))
            }
            .buttonStyle(PressScaleButtonStyle())
            .accessibilityLabel(voice.isMuted ? "Unmute microphone" : "Mute microphone")

            Spacer()

            moreMenu

            Spacer()

            Button {
                feedback()
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

    private var moreMenu: some View {
        Menu {
            if voice.hasSampleQuestions {
                Button {
                    voice.askSampleQuestion()
                } label: {
                    Label("Ask a sample question", systemImage: "waveform")
                }
            }
            Toggle(isOn: $settings.voiceCaptions) {
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

    private func close() {
        voice.end()
        router.isVoiceModePresented = false
    }

    private func feedback() {
        guard settings.haptics else { return }
        UIImpactFeedbackGenerator(style: .soft).impactOccurred()
    }
}

/// The content height of the captions, so their area hugs short captions
/// and only scrolls once they outgrow it.
private struct CaptionsHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

/// Shrinks a control slightly while pressed, without the dimming of the
/// default style, which would wash out the orb.
private struct PressScaleButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.95 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}
