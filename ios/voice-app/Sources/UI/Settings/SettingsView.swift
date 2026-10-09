import SwiftUI
import UIKit

/// Settings, as a sheet: personalization, voice, appearance, the model
/// ladder, where speech is transcribed, data controls and about.
///
/// Motion follows ChatGPT's Settings: the sheet, pushes, switches,
/// pickers and dialogs keep their system animations. What this screen adds
/// is that its own changes ease instead of snapping (a row appearing or
/// going, the preview button's glyph and words, the length counter), and
/// that switches and pickers answer with the selection tick.
///
/// Sections that read a busy controller are small views of their own:
/// `ChatController` publishes on every streamed token and dictation on
/// every audio level, and if this view observed them, a reply streaming
/// behind the sheet would redraw all of Settings many times a second,
/// enough to make the slider and typing stutter.
struct SettingsView: View {

    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var preview: VoicePreview
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @StateObject private var catalog = VoiceCatalog()

    /// About how long the system sheet takes to slide up. Work that can
    /// wait (checking the voice list) waits this long so it does not cost
    /// frames of the slide. Not a `Motion` token: it times the system's
    /// animation, not one of ours.
    private static let sheetSettleDelay: Duration = .milliseconds(500)

    var body: some View {
        NavigationStack {
            List {
                header
                personalization
                defaultMode
                voiceSection
                appSection
                ModelSection()
                SpeechRecognitionSection()
                DataControlsSection()
                about
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        // The sample stops as the sheet starts down, not
                        // after it has gone, so it does not talk over the
                        // dismissal and then cut off.
                        preview.stop()
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
        .tint(Theme.accent)
        .task {
            try? await Task.sleep(for: Self.sheetSettleDelay)
            catalog.refreshIfOpenedFromCache()
        }
        .onChange(of: scenePhase) { _, phase in
            // Back from the Settings app, where a voice may have been
            // downloaded or Personal Voice allowed.
            if phase == .active { catalog.reload() }
        }
        .onDisappear {
            // A swipe down has no earlier moment to stop the sample at.
            preview.stop()
        }
    }

    /// `binding`, plus the selection tick when the user changes it, as
    /// ChatGPT's switches and pickers answer. The setting is read after the
    /// change, so turning Haptic feedback on answers with a tick and
    /// turning it off is silent.
    private func ticking<Value>(_ binding: Binding<Value>) -> Binding<Value> {
        Binding(
            get: { binding.wrappedValue },
            set: { newValue in
                binding.wrappedValue = newValue
                Haptics.selection(enabled: settings.haptics)
            }
        )
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
            Picker("Default mode", selection: ticking($settings.defaultMode)) {
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
                VoicePickerView(catalog: catalog, selectedVoiceID: settings.voiceIdentifier)
            } label: {
                // The name crossfades and the badge fades when the voice
                // list changes; the catalog animates those changes.
                LabeledContent("Voice") {
                    HStack(spacing: 6) {
                        Text(selectedVoiceName)
                            .lineLimit(1)
                        if let voice = speakingVoice {
                            VoiceBadge(voice: voice)
                                .transition(.opacity)
                        }
                    }
                }
            }

            // Inserted and removed by the catalog inside `withMotion`, so
            // the rows below ease rather than jump when a natural voice
            // finishes downloading.
            if !catalog.hasNaturalVoice {
                NaturalVoiceHint(catalog: catalog)
            }

            SpeakingRateRow()

            previewButton

            Toggle("Voice captions", isOn: ticking($settings.voiceCaptions))
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

    /// "Preview voice" and "Stop preview" trade places with the system's
    /// symbol replace and a crossfade of the words; the row keeps its
    /// height. `isPlaying` also turns false by itself when the sample ends,
    /// outside any transaction, so the animation is keyed to the value here.
    private var previewButton: some View {
        Button {
            Haptics.tap(enabled: settings.haptics)
            preview.toggle(voiceIdentifier: settings.voiceIdentifier, rate: settings.speechRate)
        } label: {
            Label {
                Text(preview.isPlaying ? "Stop preview" : "Preview voice")
                    .contentTransition(.opacity)
            } icon: {
                Image(systemName: preview.isPlaying ? "stop.fill" : "play.fill")
                    .contentTransition(.symbolEffect(.replace))
            }
            .animation(Motion.control, value: preview.isPlaying)
        }
    }

    // MARK: - App

    private var appSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 10) {
                Text("Appearance")
                // The colours change in one frame, as ChatGPT's do; only the
                // tick is added.
                Picker("Appearance", selection: ticking($settings.appearance)) {
                    ForEach(AppSettings.Appearance.allCases) { appearance in
                        Text(appearance.title).tag(appearance)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            .padding(.vertical, 4)

            Toggle("Haptic feedback", isOn: ticking($settings.haptics))
            // Animated, so the stats lines under the replies behind the
            // sheet ease in and out instead of jumping the transcript.
            Toggle(
                "Show engine stats",
                isOn: ticking($settings.showEngineStats).animation(Motion.reduced(Motion.content, reduceMotion))
            )
        } header: {
            Text("App")
        } footer: {
            Text("Engine stats show the time to first token, the decoding speed and how many prompt tokens Pie reused from earlier turns, under each reply.")
        }
    }

    // MARK: - About

    private var about: some View {
        Section("About") {
            LabeledContent("Version", value: Self.version)
            EngineRow()
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

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The cap ChatGPT puts on each instruction box. It also keeps the
    /// system prompt a small share of the phone's 8k-token context.
    private let limit = 1500

    /// The counter shows from 200 characters before the limit.
    private var showsCounter: Bool { text.count > limit - 200 }
    private var isAtLimit: Bool { text.count >= limit }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.subheadline.weight(.semibold))
            // The hint goes the moment typing starts, like a system
            // placeholder; it is not animated.
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
            if showsCounter {
                Text("\(text.count)/\(limit)")
                    .font(.caption)
                    .foregroundStyle(isAtLimit ? Theme.destructive : Theme.secondaryInk)
                    .animation(Motion.control, value: isAtLimit)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .transition(.opacity)
            }
        }
        .padding(.vertical, 4)
        // The counter fades in as the row grows to make room for it,
        // instead of the row jumping and pushing every section below.
        .animation(Motion.reduced(Motion.content, reduceMotion), value: showsCounter)
        .onChange(of: text) { _, newValue in
            if newValue.count > limit {
                text = String(newValue.prefix(limit))
            }
        }
    }
}

/// The speaking-rate slider.
///
/// While the thumb moves, only this row redraws: the rate is saved, and so
/// reaches the rest of the app, when the finger lifts. Saving on every
/// tick re-rendered the whole app (every view that reads `AppSettings`)
/// and wrote the defaults store at the drag's rate, enough to make the
/// thumb stutter in a long chat. Then the sample plays at the new rate, as
/// before.
private struct SpeakingRateRow: View {
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var preview: VoicePreview

    /// The thumb's value while a drag is under way; nil otherwise.
    @State private var draggedRate: Float?
    @State private var isDragging = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Speaking rate")
            Slider(value: rate, in: 0.35...0.6) {
                Text("Speaking rate")
            } minimumValueLabel: {
                Image(systemName: "tortoise")
                    .foregroundStyle(Theme.secondaryInk)
            } maximumValueLabel: {
                Image(systemName: "hare")
                    .foregroundStyle(Theme.secondaryInk)
            } onEditingChanged: { editing in
                isDragging = editing
                if !editing {
                    if let draggedRate { settings.speechRate = draggedRate }
                    draggedRate = nil
                    preview.play(voiceIdentifier: settings.voiceIdentifier, rate: settings.speechRate)
                }
            }
        }
        .padding(.vertical, 4)
    }

    /// A change outside a drag (VoiceOver's adjust, for one) is saved at
    /// once, since no lift will follow to save it.
    private var rate: Binding<Float> {
        Binding(
            get: { draggedRate ?? settings.speechRate },
            set: { newRate in
                if isDragging {
                    draggedRate = newRate
                } else {
                    settings.speechRate = newRate
                }
            }
        )
    }
}

/// The model ladder.
///
/// A view of its own with no inputs, so it redraws only when it first
/// appears or its own state changes: `isPresent` lists directories on
/// disk, and inside `SettingsView` it ran on every keystroke and slider
/// tick anywhere in Settings.
///
/// Each rung carries its own confirmation, so where the system grows a
/// dialog from the control that opened it (iOS 26), it grows from the
/// tapped row.
private struct ModelSection: View {
    @State private var modelToSwitchTo: PieRuntimeConfig.Model?

    var body: some View {
        Section {
            ForEach(PieRuntimeConfig.ladder, id: \.directory) { model in
                let isInstalled = model.isPresent
                let isSelected = model == PieRuntimeConfig.selected
                Button {
                    if !isSelected { modelToSwitchTo = model }
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.label)
                                .foregroundStyle(isInstalled ? Theme.ink : Theme.tertiaryInk)
                            Text(isInstalled ? "Installed" : "Not installed")
                                .font(.caption)
                                .foregroundStyle(Theme.secondaryInk)
                        }
                        Spacer()
                        if isSelected {
                            Image(systemName: "checkmark")
                                .font(.body.weight(.semibold))
                                .foregroundStyle(Theme.accent)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .disabled(!isInstalled)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
                .confirmationDialog(
                    "Switch to \(model.label)?",
                    isPresented: confirmsSwitch(to: model),
                    titleVisibility: .visible
                ) {
                    Button("Switch and close Pie") { switchModel(to: model) }
                } message: {
                    Text("The engine loads a model once per launch, so Pie closes now and opens with the new model next time.")
                }
            }
        } header: {
            Text("Model")
        } footer: {
            Text("Switching models restarts the app.")
        }
    }

    private func confirmsSwitch(to model: PieRuntimeConfig.Model) -> Binding<Bool> {
        Binding(
            get: { modelToSwitchTo == model },
            set: { if !$0 { modelToSwitchTo = nil } }
        )
    }

    private func switchModel(to model: PieRuntimeConfig.Model) {
        PieRuntimeConfig.select(model)
        print("[settings] model switch to \(model.directory); exiting for a clean boot")
        // A moment for the dialog to close and the choice to reach the
        // defaults store before the process ends. iOS has no relaunch, and
        // the engine cannot load a second set of weights in this process.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            // Chat saves run on a background queue; let them land first.
            ChatStore.finishPendingWrites()
            exit(0)
        }
    }
}

/// Where speech is turned into text.
///
/// A view of its own because it reads the voice-mode and dictation
/// controllers, which publish often (dictation on every audio level while
/// it runs); only this section redraws for them. When the answer changes,
/// the icon is replaced with the system's symbol effect, the words and the
/// footer crossfade, and the "Allow in the Settings app" row eases in or
/// out.
private struct SpeechRecognitionSection: View {
    @EnvironmentObject private var voice: VoiceModeController
    @EnvironmentObject private var dictation: DictationController

    var body: some View {
        let availability = voice.availability ?? dictation.availability
        Section {
            // A plain row: a Label as LabeledContent's value lays out as a
            // title-and-subtitle pair and leaves the row several lines tall.
            HStack(spacing: 6) {
                Text("Transcription")
                Spacer()
                Group {
                    Image(systemName: Self.transcriptionIcon(availability))
                        .imageScale(.small)
                        .contentTransition(.symbolEffect(.replace))
                    Text(Self.transcriptionPlace(availability))
                        .contentTransition(.opacity)
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
            Text(Self.transcriptionFooter(availability))
                .contentTransition(.opacity)
        }
        .animation(Motion.crossfade, value: availability)
    }

    private static func transcriptionPlace(_ availability: VoiceInputAvailability?) -> String {
        switch availability {
        case .ready(onDevice: true)?: return "On this iPhone"
        case .ready(onDevice: false)?: return "Apple's servers"
        case .denied?: return "Not allowed"
        case .unavailable?: return "Unavailable"
        case nil: return "Not checked yet"
        }
    }

    private static func transcriptionIcon(_ availability: VoiceInputAvailability?) -> String {
        switch availability {
        case .ready(onDevice: true)?: return "lock.fill"
        case .ready(onDevice: false)?: return "icloud"
        case .denied?, .unavailable?: return "exclamationmark.triangle"
        case nil: return "questionmark.circle"
        }
    }

    private static func transcriptionFooter(_ availability: VoiceInputAvailability?) -> String {
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
}

/// "Delete all chats" and its confirmation. A view of its own so that
/// observing the chat controller, which publishes on every streamed token,
/// redraws this one row and not all of Settings. The dialog hangs on the
/// button, so where the system grows a dialog from its control (iOS 26),
/// it grows from this row.
private struct DataControlsSection: View {
    @EnvironmentObject private var store: ChatStore
    @EnvironmentObject private var chat: ChatController

    @State private var confirmsDeleteAll = false

    var body: some View {
        Section {
            Button("Delete all chats", role: .destructive) {
                confirmsDeleteAll = true
            }
            .foregroundStyle(Theme.destructive)
            .confirmationDialog(
                "Delete all chats?",
                isPresented: $confirmsDeleteAll,
                titleVisibility: .visible
            ) {
                Button("Delete all chats", role: .destructive) {
                    // Animated, so the open sidebar's rows and the chat
                    // behind the sheet fade to the empty state rather than
                    // vanishing in one frame. No haptic: ChatGPT plays none
                    // for a delete.
                    withMotion(Motion.crossfade) {
                        store.deleteAll()
                        chat.newChat()
                    }
                }
            } message: {
                Text("This removes every saved chat from this iPhone. It can't be undone.")
            }
        } header: {
            Text("Data controls")
        } footer: {
            Text("Chats are saved only on this iPhone. Temporary chats are never saved.")
        }
    }
}

/// The engine line in About. A view of its own so that observing the chat
/// controller, which publishes on every streamed token, redraws this row
/// and not all of Settings.
private struct EngineRow: View {
    @EnvironmentObject private var chat: ChatController

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Engine")
            Text(chat.engineDescription)
                .font(.subheadline)
                .foregroundStyle(Theme.secondaryInk)
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
    }
}
