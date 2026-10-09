import SwiftUI
import UIKit

/// The installed voices, best first, each with its quality. Choosing one
/// plays the preview in it, the way ChatGPT's voice picker does.
///
/// The old synthetic voices sit behind "More voices": they are what makes
/// a reply sound like a 1980s computer, and nobody should land on Fred by
/// scrolling past the good ones.
///
/// Motion, as in ChatGPT's lists: the push, the row highlight and the
/// "More voices" disclosure are the system's. Choosing a voice moves the
/// tick with a quick fade (the old one fades out as the new one fades in
/// from a little smaller) and a selection tick; while the sample plays the
/// tick turns into moving speaker waves, and back when it ends.
struct VoicePickerView: View {

    @ObservedObject var catalog: VoiceCatalog

    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var preview: VoicePreview
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var showsLegacy: Bool
    @State private var isRequestingPersonalVoice = false

    /// `selectedVoiceID` is the saved voice when the picker opens. A legacy
    /// voice chosen earlier opens "More voices" with its tick in view. It
    /// is decided here, not in `onAppear`, so the list is built open
    /// instead of expanding without animation partway through the push.
    init(catalog: VoiceCatalog, selectedVoiceID: String?) {
        self.catalog = catalog
        let selectionIsLegacy = selectedVoiceID.map { id in catalog.legacy.contains { $0.id == id } } ?? false
        _showsLegacy = State(initialValue: selectionIsLegacy)
    }

    var body: some View {
        // Worked out once per redraw and handed to every row, rather than
        // looked up again by each row.
        let selectedID = self.selectedID
        let unlisted = unlistedSelection(selectedID)

        List {
            Section {
                automaticRow(isSelected: selectedID == nil)
            }

            if let voice = unlisted {
                Section {
                    row(voice, selectedID: selectedID)
                } header: {
                    Text("Current voice")
                } footer: {
                    Text("Not one of the \(catalog.languageName) voices below. Pie keeps using it until you choose another.")
                }
            }

            if !catalog.hasNaturalVoice {
                Section {
                    NaturalVoiceHint(catalog: catalog)
                }
            }

            personalVoiceSection(selectedID: selectedID)

            if !catalog.natural.isEmpty {
                Section {
                    ForEach(catalog.natural) { voice in
                        row(voice, selectedID: selectedID)
                    }
                } header: {
                    Text("Voices")
                } footer: {
                    if catalog.hasNaturalVoice {
                        Text("Premium and Enhanced voices sound the most natural. Download more in \(SpeechSettingsPath.voices(language: catalog.languageName)).")
                    }
                }
            }

            if !catalog.legacy.isEmpty {
                Section {
                    DisclosureGroup(isExpanded: $showsLegacy) {
                        ForEach(catalog.legacy) { voice in
                            row(voice, selectedID: selectedID)
                        }
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("More voices")
                                .foregroundStyle(Theme.ink)
                            Text("Older synthetic voices. They sound robotic.")
                                .font(.caption)
                                .foregroundStyle(Theme.secondaryInk)
                        }
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Voice")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Selection

    /// The voice that will actually speak, as the picker shows it: a saved
    /// identifier that is no longer installed (a deleted download, or a
    /// Personal Voice Pie may no longer use) falls back to Automatic in the
    /// synthesizer, so it does here too.
    private var selectedID: String? {
        guard let id = settings.voiceIdentifier, catalog.voice(withID: id) != nil else { return nil }
        return id
    }

    /// A chosen voice that still speaks but is in none of the lists below,
    /// such as one picked before the iPhone's language changed. Without a
    /// row of its own nothing in the picker would carry the checkmark,
    /// while the main Settings screen names it as the voice in use.
    private func unlistedSelection(_ selectedID: String?) -> VoiceCatalog.Voice? {
        guard let id = selectedID,
              !(catalog.personal + catalog.natural + catalog.legacy).contains(where: { $0.id == id })
        else { return nil }
        return catalog.voice(withID: id)
    }

    private func choose(_ id: String?) {
        Haptics.selection(enabled: settings.haptics)
        // Animated, so the tick moves with a fade and a "Current voice"
        // section that no longer applies folds away, instead of every row
        // below it jumping up under the finger.
        withMotion(Motion.content) {
            settings.voiceIdentifier = id
        }
        preview.play(voiceIdentifier: id, rate: settings.speechRate)
    }

    // MARK: - Rows

    private func automaticRow(isSelected: Bool) -> some View {
        Button {
            choose(nil)
        } label: {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Automatic")
                        .foregroundStyle(Theme.ink)
                    Text(automaticDetail)
                        .font(.caption)
                        .foregroundStyle(Theme.secondaryInk)
                }
                Spacer()
                if let voice = catalog.automatic {
                    VoiceBadge(voice: voice)
                        .transition(.opacity)
                }
                selectionMark(isSelected)
            }
            .contentShape(Rectangle())
        }
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var automaticDetail: String {
        guard let voice = catalog.automatic else {
            return "The best voice installed for \(catalog.languageName)"
        }
        if voice.quality > .standard {
            return "Uses \(voice.name), the best voice installed for \(catalog.languageName)"
        }
        return "Uses \(voice.name) until a Premium or Enhanced voice is installed"
    }

    private func row(_ voice: VoiceCatalog.Voice, selectedID: String?) -> some View {
        let isSelected = selectedID == voice.id
        return Button {
            choose(voice.id)
        } label: {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(voice.name)
                        .foregroundStyle(Theme.ink)
                    Text(detail(voice))
                        .font(.caption)
                        .foregroundStyle(Theme.secondaryInk)
                }
                Spacer()
                VoiceBadge(voice: voice)
                selectionMark(isSelected)
            }
            .contentShape(Rectangle())
        }
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private func detail(_ voice: VoiceCatalog.Voice) -> String {
        if voice.isPersonal { return "Your Personal Voice" }
        if voice.isLegacy { return "Older synthetic voice" }
        return Locale.current.localizedString(forIdentifier: voice.language) ?? voice.language
    }

    /// The tick on the chosen voice, which turns into moving speaker waves
    /// while its sample plays, the way ChatGPT shows what is speaking.
    ///
    /// Every row keeps the slot, so nothing in a row shifts: an invisible
    /// checkmark sizes it and the visible glyph is drawn over it (the
    /// waves are a few points wider and spill evenly into the spacing).
    /// The tick fades rather than appearing in one frame; the glyph swap
    /// uses the system's symbol replace.
    private func selectionMark(_ isSelected: Bool) -> some View {
        let isPlaying = isSelected && preview.isPlaying
        return Image(systemName: "checkmark")
            .font(.body.weight(.semibold))
            .hidden()
            .overlay {
                Image(systemName: isPlaying ? "speaker.wave.2.fill" : "checkmark")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(Theme.accent)
                    .contentTransition(.symbolEffect(.replace))
                    .symbolEffect(.variableColor.iterative, isActive: isPlaying && !reduceMotion)
                    .fixedSize()
                    .animation(Motion.control, value: isPlaying)
            }
            .opacity(isSelected ? 1 : 0)
            .scaleEffect(isSelected || reduceMotion ? 1 : 0.85)
            .animation(isSelected ? Motion.fadeIn : Motion.fadeOut, value: isSelected)
            .accessibilityHidden(true)
    }

    // MARK: - Personal Voice

    @ViewBuilder
    private func personalVoiceSection(selectedID: String?) -> some View {
        switch catalog.personalVoiceAccess {
        case .unsupported:
            EmptyView()

        case .notDetermined:
            Section {
                Button(action: requestPersonalVoice) {
                    HStack(spacing: 12) {
                        Image(systemName: "person.wave.2")
                            .foregroundStyle(Theme.orange)
                            .frame(width: 24)
                            .accessibilityHidden(true)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Use my Personal Voice")
                                .foregroundStyle(Theme.ink)
                            Text("Hear replies in a voice that sounds like you, created in Accessibility settings.")
                                .font(.caption)
                                .foregroundStyle(Theme.secondaryInk)
                        }
                        Spacer()
                        if isRequestingPersonalVoice {
                            ProgressView()
                                .accessibilityLabel("Asking for permission")
                                .transition(.opacity)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .disabled(isRequestingPersonalVoice)
            } header: {
                Text("Personal Voice")
            }

        case .denied:
            Section {
                Text("Pie isn't allowed to use your Personal Voice.")
                    .foregroundStyle(Theme.ink)
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    Link("Open the Settings app", destination: url)
                }
            } header: {
                Text("Personal Voice")
            } footer: {
                Text("You can allow it in \(SpeechSettingsPath.personalVoice).")
            }

        case .authorized:
            Section {
                if catalog.personal.isEmpty && catalog.isLookingForPersonalVoice {
                    HStack(spacing: 12) {
                        ProgressView()
                        Text("Looking for your Personal Voice")
                            .font(.subheadline)
                            .foregroundStyle(Theme.secondaryInk)
                    }
                } else if catalog.personal.isEmpty {
                    Text("There's no Personal Voice on this iPhone yet. You can make one in \(SpeechSettingsPath.personalVoice).")
                        .font(.subheadline)
                        .foregroundStyle(Theme.secondaryInk)
                } else {
                    ForEach(catalog.personal) { voice in
                        row(voice, selectedID: selectedID)
                    }
                }
            } header: {
                Text("Personal Voice")
            } footer: {
                Text("A voice that sounds like you, created in \(SpeechSettingsPath.personalVoice).")
            }
        }
    }

    /// The system asks only now, on the user's tap. When a Personal Voice
    /// comes back, it is chosen and previewed, since that is what the tap
    /// asked for. The spinner fades in and out; the section's change from
    /// asking to allowed or not allowed is animated by the catalog.
    private func requestPersonalVoice() {
        withMotion(Motion.control) {
            isRequestingPersonalVoice = true
        }
        catalog.requestPersonalVoice { voice in
            withMotion(Motion.control) {
                isRequestingPersonalVoice = false
            }
            if let voice {
                choose(voice.id)
            }
        }
    }
}
