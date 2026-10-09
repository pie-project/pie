import SwiftUI
import UIKit

/// A voice's quality as a small capsule. Premium sits in the Yale band's
/// colour and Enhanced in the site's grey panel, so the two worth choosing
/// stand out; Default is only outlined, and a Personal Voice takes the
/// site's burnt orange because it is the user's own.
struct VoiceBadge: View {
    let title: String
    let style: Style

    enum Style {
        case premium
        case enhanced
        case standard
        case personal
    }

    init(quality: VoiceCatalog.Quality) {
        switch quality {
        case .premium: self.init(title: quality.title, style: .premium)
        case .enhanced: self.init(title: quality.title, style: .enhanced)
        case .standard: self.init(title: quality.title, style: .standard)
        }
    }

    init(voice: VoiceCatalog.Voice) {
        if voice.isPersonal {
            self.init(title: "Personal", style: .personal)
        } else {
            self.init(quality: voice.quality)
        }
    }

    private init(title: String, style: Style) {
        self.title = title
        self.style = style
    }

    var body: some View {
        Text(title)
            .font(.caption2.weight(.bold))
            .foregroundStyle(foreground)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Capsule().fill(fill))
            .overlay(Capsule().strokeBorder(style == .standard ? Theme.hairline : .clear, lineWidth: 1))
            .fixedSize()
    }

    private var foreground: Color {
        switch style {
        case .premium, .personal: return Theme.onAccent
        case .enhanced: return Theme.ink
        case .standard: return Theme.secondaryInk
        }
    }

    private var fill: Color {
        switch style {
        case .premium: return Theme.accentFill
        case .personal: return Theme.orange
        case .enhanced: return Theme.surfaceStrong
        case .standard: return .clear
        }
    }
}

/// Where the voice downloads live in the Settings app. iOS 26 renamed
/// Accessibility's "Spoken Content" to "Read & Speak"; the Voices list
/// inside it is the same.
enum SpeechSettingsPath {
    static var spokenContent: String {
        if #available(iOS 26, *) {
            return "Read & Speak"
        }
        return "Spoken Content"
    }

    static func voices(language: String) -> String {
        "Settings > Accessibility > \(spokenContent) > Voices > \(language)"
    }

    static let personalVoice = "Settings > Accessibility > Personal Voice"
}

/// Shown while only compact voices are installed: what to download, where,
/// and that it is a one-time download that then works offline. There is no
/// public link into Accessibility settings, so the button opens Pie's own
/// page in the Settings app. Since iOS 18 that page sits under Apps, two
/// Backs from the top level, so the caption says where to go rather than
/// how many times to tap Back.
struct NaturalVoiceHint: View {
    let catalog: VoiceCatalog

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label {
                Text("Get a more natural voice")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.ink)
            } icon: {
                Image(systemName: "waveform")
                    .foregroundStyle(Theme.accent)
            }
            Text(message)
                .font(.subheadline)
                .foregroundStyle(Theme.secondaryInk)
                .fixedSize(horizontal: false, vertical: true)
            if let url = URL(string: UIApplication.openSettingsURLString) {
                VStack(alignment: .leading, spacing: 2) {
                    Link(destination: url) {
                        Text("Open the Settings app")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(Theme.accent)
                    }
                    .buttonStyle(.borderless)
                    Text("It opens on Pie's page. Go back to the top of Settings, then tap Accessibility.")
                        .font(.caption)
                        .foregroundStyle(Theme.tertiaryInk)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var message: String {
        let path = SpeechSettingsPath.voices(language: catalog.languageName)
        let which = catalog.suggestsAvaAndZoe
            ? "Ava (Premium) or Zoe (Premium)"
            : "a voice for your region marked Premium or Enhanced"
        return "Only basic voices are installed, which is why Pie can sound robotic. "
            + "Go to \(path) and download \(which). "
            + "It's a one-time download, and the voice works offline."
    }
}
