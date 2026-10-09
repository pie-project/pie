import SwiftUI
import UIKit

/// Settings, as a sheet: personalization, voice, appearance, the model
/// ladder, where speech is transcribed, data controls and about.
struct SettingsView: View {

    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var store: ChatStore
    @EnvironmentObject private var chat: ChatController
    @EnvironmentObject private var voice: VoiceModeController
    @EnvironmentObject private var dictation: DictationController
    @EnvironmentObject private var preview: VoicePreview
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase

    @StateObject private var catalog = VoiceCatalog()
    @State private var confirmsDeleteAll = false
    @State private var modelToSwitchTo: PieRuntimeConfig.Model?

    var body: some View {
        NavigationStack {
            List {
                header
                personalization
                defaultMode
                voiceSection
                appSection
                modelSection
                speechRecognition
                dataControls
                about
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .fontWeight(.semibold)
                }
            }
            .confirmationDialog(
                "Delete all chats?",
                isPresented: $confirmsDeleteAll,
                titleVisibility: .visible
            ) {
                Button("Delete all chats", role: .destructive) {
                    store.deleteAll()
                    chat.newChat()
                }
            } message: {
                Text("This removes every saved chat from this iPhone. It can't be undone.")
            }
            .confirmationDialog(
                modelToSwitchTo.map { "Switch to \($0.label)?" } ?? "",
                isPresented: Binding(
                    get: { modelToSwitchTo != nil },
                    set: { if !$0 { modelToSwitchTo = nil } }
                ),
                titleVisibility: .visible,
                presenting: modelToSwitchTo
            ) { model in
                Button("Switch and close Pie") { switchModel(to: model) }
            } message: { _ in
                Text("The engine loads a model once per launch, so Pie closes now and opens with the new model next time.")
            }
        }
        .tint(Theme.accent)
        .onChange(of: scenePhase) { _, phase in
            // Back from the Settings app, where a voice may have been
            // downloaded or Personal Voice allowed.
            if phase == .active { catalog.reload() }
        }
        .onDisappear {
            preview.stop()
        }
    }

    // MARK: - Header

    private var header: some View {
        Section {
            VStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Pie Voice")
                        .font(Theme.serif(28))
                        .foregroundStyle(Theme.onTopBar)
                    Text("\(PieRuntimeConfig.pieVersion) · \(PieRuntimeConfig.driverDescription) · \(PieRuntimeConfig.modelDescription) · runs entirely on this iPhone")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.onTopBar.opacity(0.85))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 20)
                .padding(.vertical, 18)

                // The site's grey rule under its blue band.
                Rectangle()
                    .fill(Theme.band)
                    .frame(height: 4)
            }
            .listRowInsets(EdgeInsets())
            .listRowBackground(Theme.topBar)
            .accessibilityElement(children: .combine)
        }
    }

    // MARK: - Personalization

    private var personalization: some View {
        Section {
            InstructionField(
                title: "What should Pie know about you?",
                placeholder: "Your name, what you study or do, what you're into, where you live…",
                text: $settings.aboutYou
            )
            InstructionField(
                title: "How should Pie respond?",
                placeholder: "Short and direct, explain like I'm new to it, use examples, be encouraging…",
                text: $settings.responseTraits
            )
        } header: {
            Text("Personalization")
        } footer: {
            Text("Given to Pie at the start of every chat. Like everything else here, it stays on this iPhone.")
        }
    }

    // MARK: - Default mode

    private var defaultMode: some View {
        Section {
            Picker("Default mode", selection: $settings.defaultMode) {
                ForEach(ReplyMode.allCases) { mode in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(mode.title)
                        Text(mode.subtitle)
                            .font(.caption)
                            .foregroundStyle(Theme.secondaryInk)
                    }
                    .tag(mode)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
        } header: {
            Text("Default mode")
        } footer: {
            Text("New chats start in this mode. Switch any time from the composer.")
        }
    }

    // MARK: - Voice

    private var voiceSection: some View {
        Section {
            NavigationLink {
                VoicePickerView(catalog: catalog)
            } label: {
                LabeledContent("Voice") {
                    HStack(spacing: 6) {
                        Text(selectedVoiceName)
                            .lineLimit(1)
                        if let voice = speakingVoice {
                            VoiceBadge(voice: voice)
                        }
                    }
                }
            }

            if !catalog.hasNaturalVoice {
                NaturalVoiceHint(catalog: catalog)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Speaking rate")
                Slider(value: $settings.speechRate, in: 0.35...0.6) {
                    Text("Speaking rate")
                } minimumValueLabel: {
                    Image(systemName: "tortoise")
                        .foregroundStyle(Theme.secondaryInk)
                } maximumValueLabel: {
                    Image(systemName: "hare")
                        .foregroundStyle(Theme.secondaryInk)
                } onEditingChanged: { editing in
                    if !editing { previewSelectedVoice() }
                }
            }
            .padding(.vertical, 4)

            Button {
                preview.toggle(voiceIdentifier: settings.voiceIdentifier, rate: settings.speechRate)
            } label: {
                Label(
                    preview.isPlaying ? "Stop preview" : "Preview voice",
                    systemImage: preview.isPlaying ? "stop.fill" : "play.fill"
                )
            }

            Toggle("Voice captions", isOn: $settings.voiceCaptions)
        } header: {
            Text("Voice")
        } footer: {
            Text("Captions show what you said and Pie's reply as text in voice mode.")
        }
    }

    /// The chosen voice, or nil when Automatic is in force, including when
    /// the saved voice is no longer installed and the synthesizer falls
    /// back to Automatic.
    private var chosenVoice: VoiceCatalog.Voice? {
        settings.voiceIdentifier.flatMap(catalog.voice(withID:))
    }

    /// The voice replies are spoken in right now.
    private var speakingVoice: VoiceCatalog.Voice? {
        chosenVoice ?? catalog.automatic
    }

    /// "Automatic (Ava)" says which voice Automatic resolves to, so a
    /// robotic-sounding reply can be traced to the voice behind it.
    private var selectedVoiceName: String {
        if let chosenVoice { return chosenVoice.name }
        guard let automatic = catalog.automatic else { return "Automatic" }
        return "Automatic (\(automatic.name))"
    }

    private func previewSelectedVoice() {
        preview.play(voiceIdentifier: settings.voiceIdentifier, rate: settings.speechRate)
    }

    // MARK: - App

    private var appSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 10) {
                Text("Appearance")
                Picker("Appearance", selection: $settings.appearance) {
                    ForEach(AppSettings.Appearance.allCases) { appearance in
                        Text(appearance.title).tag(appearance)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            .padding(.vertical, 4)

            Toggle("Haptic feedback", isOn: $settings.haptics)
            Toggle("Show engine stats", isOn: $settings.showEngineStats)
        } header: {
            Text("App")
        } footer: {
            Text("Engine stats show the time to first token, the decoding speed and how many prompt tokens Pie reused from earlier turns, under each reply.")
        }
    }

    // MARK: - Model

    private var modelSection: some View {
        Section {
            ForEach(PieRuntimeConfig.ladder, id: \.directory) { model in
                Button {
                    if model != PieRuntimeConfig.selected { modelToSwitchTo = model }
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.label)
                                .foregroundStyle(model.isPresent ? Theme.ink : Theme.tertiaryInk)
                            Text(model.isPresent ? "Installed" : "Not installed")
                                .font(.caption)
                                .foregroundStyle(Theme.secondaryInk)
                        }
                        Spacer()
                        if model == PieRuntimeConfig.selected {
                            Image(systemName: "checkmark")
                                .font(.body.weight(.semibold))
                                .foregroundStyle(Theme.accent)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .disabled(!model.isPresent)
                .accessibilityAddTraits(model == PieRuntimeConfig.selected ? .isSelected : [])
            }
        } header: {
            Text("Model")
        } footer: {
            Text("Switching models restarts the app.")
        }
    }

    private func switchModel(to model: PieRuntimeConfig.Model) {
        PieRuntimeConfig.select(model)
        print("[settings] model switch to \(model.directory); exiting for a clean boot")
        // A moment for the dialog to close and the choice to reach the
        // defaults store before the process ends. iOS has no relaunch, and
        // the engine cannot load a second set of weights in this process.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            exit(0)
        }
    }

    // MARK: - Speech recognition

    private var speechRecognition: some View {
        let availability = voice.availability ?? dictation.availability
        return Section {
            // A plain row: a Label as LabeledContent's value lays out as a
            // title-and-subtitle pair and leaves the row several lines tall.
            HStack(spacing: 6) {
                Text("Transcription")
                Spacer()
                Group {
                    Image(systemName: transcriptionIcon(availability))
                        .imageScale(.small)
                    Text(transcriptionPlace(availability))
                }
                .foregroundStyle(Theme.secondaryInk)
            }
            .accessibilityElement(children: .combine)
            if case .denied? = availability, let url = URL(string: UIApplication.openSettingsURLString) {
                Link("Allow in the Settings app", destination: url)
            }
        } header: {
            Text("Speech recognition")
        } footer: {
            Text(transcriptionFooter(availability))
        }
    }

    private func transcriptionPlace(_ availability: VoiceInputAvailability?) -> String {
        switch availability {
        case .ready(onDevice: true)?: return "On this iPhone"
        case .ready(onDevice: false)?: return "Apple's servers"
        case .denied?: return "Not allowed"
        case .unavailable?: return "Unavailable"
        case nil: return "Not checked yet"
        }
    }

    private func transcriptionIcon(_ availability: VoiceInputAvailability?) -> String {
        switch availability {
        case .ready(onDevice: true)?: return "lock.fill"
        case .ready(onDevice: false)?: return "icloud"
        case .denied?, .unavailable?: return "exclamationmark.triangle"
        case nil: return "questionmark.circle"
        }
    }

    private func transcriptionFooter(_ availability: VoiceInputAvailability?) -> String {
        switch availability {
        case .ready(onDevice: true)?:
            return "What you say is turned into text on this iPhone and never leaves it."
        case .ready(onDevice: false)?:
            return "This iPhone can't transcribe your language on the device, so your speech goes to Apple's servers to be turned into text. The model itself still runs here."
        case .denied(let reason)?:
            return reason
        case .unavailable(let reason)?:
            return reason
        case nil:
            return "Checked the first time you use the microphone."
        }
    }

    // MARK: - Data controls

    private var dataControls: some View {
        Section {
            Button("Delete all chats", role: .destructive) {
                confirmsDeleteAll = true
            }
            .foregroundStyle(Theme.destructive)
        } header: {
            Text("Data controls")
        } footer: {
            Text("Chats are saved only on this iPhone. Temporary chats are never saved.")
        }
    }

    // MARK: - About

    private var about: some View {
        Section("About") {
            LabeledContent("Version", value: Self.version)
            VStack(alignment: .leading, spacing: 4) {
                Text("Engine")
                Text(chat.engineDescription)
                    .font(.subheadline)
                    .foregroundStyle(Theme.secondaryInk)
            }
            .padding(.vertical, 2)
            .accessibilityElement(children: .combine)
            if let url = URL(string: "https://github.com/aarushkandukoori/pie/tree/ios-voice-05") {
                Link(destination: url) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Source")
                            .foregroundStyle(Theme.ink)
                        Text("github.com/aarushkandukoori/pie, branch ios-voice-05")
                            .font(.subheadline)
                            .foregroundStyle(Theme.accent)
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    private static var version: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "-"
        guard let build = info?["CFBundleVersion"] as? String else { return short }
        return "\(short) (\(build))"
    }
}

/// One custom-instruction box: the question, a multi-line editor with a
/// hint while it is empty, and a length counter near the limit.
private struct InstructionField: View {
    let title: String
    let placeholder: String
    @Binding var text: String

    /// The cap ChatGPT puts on each instruction box. It also keeps the
    /// system prompt a small share of the phone's 8k-token context.
    private let limit = 1500

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.subheadline.weight(.semibold))
            ZStack(alignment: .topLeading) {
                TextEditor(text: $text)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 88)
                if text.isEmpty {
                    Text(placeholder)
                        .foregroundStyle(Theme.tertiaryInk)
                        .padding(.top, 8)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
            if text.count > limit - 200 {
                Text("\(text.count)/\(limit)")
                    .font(.caption)
                    .foregroundStyle(text.count >= limit ? Theme.destructive : Theme.secondaryInk)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .padding(.vertical, 4)
        .onChange(of: text) { _, newValue in
            if newValue.count > limit {
                text = String(newValue.prefix(limit))
            }
        }
    }
}
