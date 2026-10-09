import AVFoundation
import Combine
import Foundation

/// The voices the Settings screen offers, sorted the way a person should
/// meet them: natural-sounding voices first, the old synthetic ones (Fred,
/// Kathy, Eddy and friends) set apart, and the user's own Personal Voice
/// once they have let Pie use it.
///
/// The order, the legacy flag and the "Automatic" voice all come from
/// `SpeechSynthesis`, the code that actually speaks, so what this screen
/// says is what plays.
@MainActor
final class VoiceCatalog: ObservableObject {

    struct Voice: Identifiable, Equatable {
        /// The `AVSpeechSynthesisVoice` identifier.
        let id: String
        let name: String
        let quality: Quality
        let language: String
        /// A MacinTalk, novelty or Eloquence voice: formant synthesis from
        /// decades ago, kept for accessibility and compatibility. It is
        /// what "robotic" sounds like, so it is never offered up front.
        let isLegacy: Bool
        let isPersonal: Bool
    }

    enum Quality: Int, Comparable {
        case standard = 1
        case enhanced = 2
        case premium = 3

        init(_ name: String) {
            switch name {
            case "Premium": self = .premium
            case "Enhanced": self = .enhanced
            default: self = .standard
            }
        }

        init(_ quality: AVSpeechSynthesisVoiceQuality) {
            switch quality {
            case .premium: self = .premium
            case .enhanced: self = .enhanced
            default: self = .standard
            }
        }

        var title: String {
            switch self {
            case .premium: return "Premium"
            case .enhanced: return "Enhanced"
            case .standard: return "Default"
            }
        }

        static func < (lhs: Quality, rhs: Quality) -> Bool { lhs.rawValue < rhs.rawValue }
    }

    /// Premium, then Enhanced, then the system's own voice for the
    /// language, then the other modern Default voices: the order
    /// Automatic prefers them in.
    @Published private(set) var natural: [Voice] = []
    /// The old synthetic voices, by name, for whoever really wants one.
    @Published private(set) var legacy: [Voice] = []
    /// Only filled once Pie may use Personal Voice.
    @Published private(set) var personal: [Voice] = []
    /// What "Automatic" speaks with right now.
    @Published private(set) var automatic: Voice?
    @Published private(set) var personalVoiceAccess: PersonalVoiceAccess = .unsupported
    /// True from the moment Personal Voice is allowed until the voice is
    /// listed or the wait gives up, so the screen shows a spinner instead
    /// of saying there is no Personal Voice.
    @Published private(set) var isLookingForPersonalVoice = false

    private var voicesDidChange: AnyCancellable?
    private var personalVoiceWait: AnyCancellable?

    /// How long to wait for a just-allowed Personal Voice to be listed.
    /// Long enough for the notification that lists it, short enough that
    /// someone who has never made one is not left watching a spinner.
    private static let personalVoiceListingTimeout: TimeInterval = 3

    init() {
        reload()
        // Posted when a voice finishes downloading in the Settings app or a
        // Personal Voice becomes available, so the list and the hint keep
        // up without reopening the screen.
        voicesDidChange = NotificationCenter.default
            .publisher(for: AVSpeechSynthesizer.availableVoicesDidChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.reload() }
    }

    /// True when at least one Premium or Enhanced voice is installed for
    /// the language. Without one, every choice is a compact voice and the
    /// hint to download one is the single biggest improvement on offer.
    var hasNaturalVoice: Bool {
        natural.contains { $0.quality > .standard }
    }

    /// The user's language as the Settings app lists it under Voices, for
    /// example "English".
    var languageName: String {
        let code = AVSpeechSynthesisVoice.currentLanguageCode()
        let base = String(code.prefix { $0 != "-" && $0 != "_" })
        return Locale.current.localizedString(forLanguageCode: base) ?? code
    }

    /// Ava and Zoe are American English voices; elsewhere the hint names
    /// the quality to look for rather than a voice Automatic would skip.
    var suggestsAvaAndZoe: Bool {
        AVSpeechSynthesisVoice.currentLanguageCode() == "en-US"
    }

    /// The voice an identifier names, or nil when the synthesizer would not
    /// use it either (a deleted download, or a Personal Voice Pie may no
    /// longer use), in which case it speaks with Automatic instead.
    func voice(withID id: String) -> Voice? {
        if let listed = (personal + natural + legacy).first(where: { $0.id == id }) {
            return listed
        }
        // A voice chosen before the iPhone's language changed is still
        // installed and still what speaks, though it is not listed.
        guard let voice = AVSpeechSynthesisVoice(identifier: id) else { return nil }
        let isPersonal = voice.voiceTraits.contains(.isPersonalVoice)
        if isPersonal && personalVoiceAccess != .authorized { return nil }
        return Voice(
            id: voice.identifier,
            name: voice.name,
            quality: Quality(voice.quality),
            language: voice.language,
            isLegacy: SpeechSynthesis.isLegacyVoice(voice) || Self.isEloquence(voice.identifier),
            isPersonal: isPersonal
        )
    }

    func reload() {
        personalVoiceAccess = SpeechSynthesis.personalVoiceAccess

        let available = SpeechSynthesis.availableVoices().filter { !$0.isPersonalVoice }
        let listed = available.map(Self.voice)
        natural = listed.filter { !$0.isLegacy }
        legacy = listed.filter(\.isLegacy).sorted { ($0.name, $0.id) < ($1.name, $1.id) }
        personal = SpeechSynthesis.personalVoices().map(Self.voice)

        // `SpeechSynthesis` speaks with the first voice it does not flag as
        // legacy, in this same order, or with the system's own voice for
        // the language when there is none. Eloquence voices are shown with
        // the legacy ones here but can still be that first voice.
        let language = AVSpeechSynthesisVoice.currentLanguageCode()
        automatic = available.first { !$0.isLegacy }.map(Self.voice)
            ?? AVSpeechSynthesisVoice(language: language).flatMap { voice(withID: $0.identifier) }
    }

    /// Asks for Personal Voice, only ever on the user's tap. The completion
    /// runs on the main actor with the voice to switch to, if there is one.
    func requestPersonalVoice(completion: @escaping @MainActor (Voice?) -> Void) {
        Task { @MainActor [weak self] in
            let access = await SpeechSynthesis.requestPersonalVoiceAccess()
            guard let self else { return }
            self.reload()
            guard access == .authorized, self.personal.isEmpty else {
                completion(self.personal.first)
                return
            }
            self.waitForPersonalVoice(completion: completion)
        }
    }

    /// A Personal Voice the user has just allowed is usually listed only a
    /// moment later, when `availableVoicesDidChangeNotification` arrives.
    /// Reloading on each notification until it shows up (or the timeout
    /// passes, for someone who has not made one) lets the tap still choose
    /// and preview it instead of only listing it.
    private func waitForPersonalVoice(completion: @escaping @MainActor (Voice?) -> Void) {
        isLookingForPersonalVoice = true
        let changes = NotificationCenter.default
            .publisher(for: AVSpeechSynthesizer.availableVoicesDidChangeNotification)
            .map { _ in false }
        let deadline = Just(true)
            .delay(for: .seconds(Self.personalVoiceListingTimeout), scheduler: DispatchQueue.main)
        personalVoiceWait = changes
            .merge(with: deadline)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] timedOut in
                guard let self else { return }
                self.reload()
                guard timedOut || !self.personal.isEmpty else { return }
                self.personalVoiceWait = nil
                self.isLookingForPersonalVoice = false
                completion(self.personal.first)
            }
    }

    // MARK: - Mapping

    private static func voice(_ info: SpeechVoiceInfo) -> Voice {
        Voice(
            id: info.id,
            name: info.name,
            quality: Quality(info.quality),
            language: info.language,
            isLegacy: info.isLegacy || isEloquence(info.id),
            isPersonal: info.isPersonalVoice
        )
    }

    /// Eloquence (Eddy, Flo, Grandma, Grandpa, Reed, Rocko, Sandy,
    /// Shelley) is the screen-reader synthesizer: clear at speed, but as
    /// synthetic as Fred to anyone expecting a natural voice, so the picker
    /// keeps it with the legacy voices.
    private static func isEloquence(_ identifier: String) -> Bool {
        identifier.hasPrefix("com.apple.eloquence.")
    }
}
