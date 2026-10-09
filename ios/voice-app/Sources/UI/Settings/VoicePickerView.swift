import SwiftUI
import UIKit

/// The installed voices, best first, each with its quality. Choosing one
/// plays the preview in it, the way ChatGPT's voice picker does.
///
/// The old synthetic voices sit behind "More voices": they are what makes
/// a reply sound like a 1980s computer, and nobody should land on Fred by
/// scrolling past the good ones.
struct VoicePickerView: View {

    @ObservedObject var catalog: VoiceCatalog

    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var preview: VoicePreview

    @State private var showsLegacy = false
    @State private var isRequestingPersonalVoice = false

    var body: some View {
        List {
            Section {
                automaticRow
            }

            if let voice = unlistedSelection {
                Section {
                    row(voice)
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

            personalVoiceSection

            if !catalog.natural.isEmpty {
                Section {
                    ForEach(catalog.natural) { voice in
                        row(voice)
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
                            row(voice)
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
        .onAppear {
            // A legacy voice chosen earlier stays visible with its tick.
            if let id = selectedID, catalog.legacy.contains(where: { $0.id == id }) {
                showsLegacy = true
            }
        }
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
    private var unlistedSelection: VoiceCatalog.Voice? {
        guard let id = selectedID,
              !(catalog.personal + catalog.natural + catalog.legacy).contains(where: { $0.id == id })
        else { return nil }
        return catalog.voice(withID: id)
    }

    private func choose(_ id: String?) {
        settings.voiceIdentifier = id
        preview.play(voiceIdentifier: id, rate: settings.speechRate)
    }

    // MARK: - Rows

    private var automaticRow: some View {
        let isSelected = selectedID == nil
        return Button {
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
                }
                checkmark(isSelected)
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

    private func row(_ voice: VoiceCatalog.Voice) -> some View {
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
                checkmark(isSelected)
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

    private func checkmark(_ isSelected: Bool) -> some View {
        Image(systemName: "checkmark")
            .font(.body.weight(.semibold))
            .foregroundStyle(Theme.accent)
            .opacity(isSelected ? 1 : 0)
            .accessibilityHidden(true)
    }

    // MARK: - Personal Voice

    @ViewBuilder
    private var personalVoiceSection: some View {
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
                        row(voice)
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
    /// asked for.
    private func requestPersonalVoice() {
        isRequestingPersonalVoice = true
        catalog.requestPersonalVoice { voice in
            isRequestingPersonalVoice = false
            if let voice {
                choose(voice.id)
            }
        }
    }
}
