import Foundation
import SwiftUI

/// Everything the Settings screen changes, persisted in `UserDefaults`.
///
/// Read by the controllers (system prompt, voice, default mode) and the
/// views (appearance, haptics, engine stats). Nothing here is
/// Pie-specific.
final class AppSettings: ObservableObject {

    enum Appearance: String, CaseIterable, Identifiable {
        case system
        case light
        case dark

        var id: String { rawValue }

        var title: String {
            switch self {
            case .system: return "System"
            case .light: return "Light"
            case .dark: return "Dark"
            }
        }

        var colorScheme: ColorScheme? {
            switch self {
            case .system: return nil
            case .light: return .light
            case .dark: return .dark
            }
        }
    }

    private let defaults: UserDefaults

    @Published var appearance: Appearance {
        didSet { defaults.set(appearance.rawValue, forKey: Keys.appearance) }
    }
    @Published var haptics: Bool {
        didSet { defaults.set(haptics, forKey: Keys.haptics) }
    }
    /// Time to first token, decode rate and reused tokens under each reply.
    @Published var showEngineStats: Bool {
        didSet { defaults.set(showEngineStats, forKey: Keys.showEngineStats) }
    }
    /// Live captions in voice mode.
    @Published var voiceCaptions: Bool {
        didSet { defaults.set(voiceCaptions, forKey: Keys.voiceCaptions) }
    }
    /// An `AVSpeechSynthesisVoice` identifier; nil picks the best installed.
    @Published var voiceIdentifier: String? {
        didSet { defaults.set(voiceIdentifier, forKey: Keys.voiceIdentifier) }
    }
    /// `AVSpeechUtterance` rate units, 0...1.
    @Published var speechRate: Float {
        didSet { defaults.set(speechRate, forKey: Keys.speechRate) }
    }
    /// "What should Pie know about you?"
    @Published var aboutYou: String {
        didSet { defaults.set(aboutYou, forKey: Keys.aboutYou) }
    }
    /// "How should Pie respond?"
    @Published var responseTraits: String {
        didSet { defaults.set(responseTraits, forKey: Keys.responseTraits) }
    }
    /// The mode a new chat starts in.
    @Published var defaultMode: ReplyMode {
        didSet { defaults.set(defaultMode.rawValue, forKey: Keys.defaultMode) }
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        appearance = defaults.string(forKey: Keys.appearance)
            .flatMap(Appearance.init(rawValue:)) ?? .system
        haptics = defaults.object(forKey: Keys.haptics) as? Bool ?? true
        showEngineStats = defaults.object(forKey: Keys.showEngineStats) as? Bool ?? true
        voiceCaptions = defaults.object(forKey: Keys.voiceCaptions) as? Bool ?? true
        voiceIdentifier = defaults.string(forKey: Keys.voiceIdentifier)
        speechRate = defaults.object(forKey: Keys.speechRate) as? Float ?? 0.5
        aboutYou = defaults.string(forKey: Keys.aboutYou) ?? ""
        responseTraits = defaults.string(forKey: Keys.responseTraits) ?? ""
        defaultMode = defaults.string(forKey: Keys.defaultMode)
            .flatMap(ReplyMode.init(rawValue:)) ?? .instant
    }

    /// The user's custom instructions as one block for the system prompt,
    /// or empty when they have written none.
    var customInstructions: String {
        var parts: [String] = []
        let about = aboutYou.trimmingCharacters(in: .whitespacesAndNewlines)
        let traits = responseTraits.trimmingCharacters(in: .whitespacesAndNewlines)
        if !about.isEmpty { parts.append("About the user: \(about)") }
        if !traits.isEmpty { parts.append("How to respond: \(traits)") }
        return parts.joined(separator: "\n")
    }

    private enum Keys {
        static let appearance = "settings.appearance"
        static let haptics = "settings.haptics"
        static let showEngineStats = "settings.showEngineStats"
        static let voiceCaptions = "settings.voiceCaptions"
        static let voiceIdentifier = "settings.voiceIdentifier"
        static let speechRate = "settings.speechRate"
        static let aboutYou = "settings.aboutYou"
        static let responseTraits = "settings.responseTraits"
        static let defaultMode = "settings.defaultMode"
    }
}
