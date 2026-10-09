import SwiftUI

/// The installed voices, best quality first. Choosing one plays the
/// preview in it, the way ChatGPT's voice picker does.
struct VoicePickerView: View {

    let voices: [SpeechVoiceInfo]

    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var preview: VoicePreview

    var body: some View {
        List {
            Section {
                row(
                    id: nil,
                    name: "Automatic",
                    detail: "The best voice installed for your language",
                    quality: nil
                )
            }
            Section {
                ForEach(voices) { voice in
                    row(id: voice.id, name: voice.name, detail: voice.language, quality: voice.quality)
                }
            } footer: {
                Text("Download more voices in Settings > Accessibility > Spoken Content > Voices. Enhanced and Premium voices sound the most natural.")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Voice")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func row(id: String?, name: String, detail: String, quality: String?) -> some View {
        let isSelected = settings.voiceIdentifier == id
        return Button {
            settings.voiceIdentifier = id
            preview.play()
        } label: {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(name)
                        .foregroundStyle(Theme.ink)
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(Theme.secondaryInk)
                }
                Spacer()
                if let quality, quality != "Default" {
                    QualityBadge(quality: quality)
                }
                Image(systemName: "checkmark")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(Theme.accent)
                    .opacity(isSelected ? 1 : 0)
            }
            .contentShape(Rectangle())
        }
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

/// "Premium" in the Yale band's colours, "Enhanced" in the site's grey.
private struct QualityBadge: View {
    let quality: String

    var body: some View {
        Text(quality)
            .font(.caption2.weight(.bold))
            .foregroundStyle(quality == "Premium" ? Theme.onAccent : Theme.ink)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(
                Capsule().fill(quality == "Premium" ? Theme.accentFill : Theme.surfaceStrong)
            )
    }
}
