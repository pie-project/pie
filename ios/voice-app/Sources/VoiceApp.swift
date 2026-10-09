import Combine
import SwiftUI
import UIKit

/// Composition root.
///
/// The only place where the concrete engine, microphone, sample player and
/// synthesizer are named. Everything downstream sees protocols and the
/// controllers built on them, which keeps a Pie upgrade confined to
/// `PieKit/` and an audio change confined to `AudioKit/`.
@MainActor
final class AppComposition: ObservableObject {

    let settings: AppSettings
    let store: ChatStore
    let router: AppRouter
    /// Exposed so benchmark mode and the audio self-check can drive the
    /// backend without the chat in the way.
    let engine: PieEngine
    /// The one synthesizer: chat read-aloud, voice mode and the Settings
    /// preview all speak through it, so they share the engine whose echo
    /// canceller voice mode depends on.
    let speech: SpeechSynthesis
    let microphone: MicrophoneInput
    let sample: SampleQuestionInput
    let chat: ChatController
    let voice: VoiceModeController
    let dictation: DictationController
    let voicePreview: VoicePreview

    /// Turns 2 and 3 are follow-ups on purpose: they are only coherent if
    /// the backend kept the conversation, so asking the samples in a row
    /// exercises the cached-prefix path without anyone speaking.
    static let sampleResources = [
        "sample-question-1",
        "sample-question-2",
        "sample-question-3",
    ]

    private var subscriptions: Set<AnyCancellable> = []
    private var didLaunch = false

    init() {
        // Restore the ladder rung chosen on a previous launch before
        // anything reads the engine description or logs the model, and
        // mirror the console before anything prints.
        PieRuntimeConfig.restoreSelection()
        ConsoleMirror.install()
        LaunchDiagnostics.log()

        settings = AppSettings()
        store = ChatStore()
        router = AppRouter()
        engine = PieEngine()
        speech = SpeechSynthesis()
        microphone = MicrophoneInput()
        sample = SampleQuestionInput(resources: Self.sampleResources)
        // `-PieDemoBackend 1` swaps in canned, paced replies so the
        // interface can be exercised where the engine cannot run.
        let backend: ConversationBackend = DemoBackend.isEnabled ? DemoBackend() : engine
        chat = ChatController(store: store, backend: backend, speech: speech, settings: settings)
        // A build without the recordings simply has no sample question;
        // voice mode hides the menu item rather than failing on tap.
        voice = VoiceModeController(
            chat: chat,
            microphone: microphone,
            sample: sample.hasRecordings ? sample : nil,
            speech: speech,
            settings: settings
        )
        dictation = DictationController(microphone: microphone)
        voicePreview = VoicePreview(speech: speech, chat: chat)

        // `@Published` replays its current value on subscription, so these
        // also apply the saved voice and rate before anything is spoken.
        settings.$voiceIdentifier
            .sink { [speech] identifier in speech.voiceIdentifier = identifier }
            .store(in: &subscriptions)
        settings.$speechRate
            .sink { [speech] rate in speech.rate = rate }
            .store(in: &subscriptions)
    }

    /// Runs once, when the first window appears. The launch arguments pick
    /// exactly one of the modes; each harness drives the engine itself, so
    /// the chat's warm-up would race it.
    func launch() {
        guard !didLaunch else { return }
        didLaunch = true
        // Learns how this device hides the keyboard from the first hide
        // of any kind, before the first send needs to know.
        KeyboardDismissal.startListening()

        if BenchmarkRunner.isEnabled {
            BenchmarkRunner.run(backend: engine)
        } else if AudioSelfCheck.isEnabled {
            AudioSelfCheck.run(backend: engine, speech: speech, microphone: microphone, sample: sample)
        } else {
            chat.bootstrap()
            if UITour.isEnabled {
                UITour.run(chat: chat, voice: voice, router: router, settings: settings, store: store)
            } else if VoiceSoak.isEnabled {
                VoiceSoak.run(
                    chat: chat,
                    voice: voice,
                    speech: speech,
                    microphone: microphone,
                    router: router,
                    store: store
                )
            }
        }
    }
}

/// Mirrors stdout/stderr into a file inside the app container.
///
/// On a physical device there is no attached console, and devicectl's
/// console capture has proven unreliable — so everything the engine
/// prints (boot and weight-load markers, sandbox pool sizing, Rust
/// panics) goes to `Documents/pie-console.log`, which can be pulled off the
/// phone afterwards. It is what caught the 4 TB address-space panic
/// that no Swift `catch` could ever see.
enum ConsoleMirror {
    static var path: String {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("pie-console.log").path
    }

    static func install() {
        #if !targetEnvironment(simulator)
        // Benchmark runs are read off the devicectl console by
        // bench-ladder.sh; redirecting stdout would leave it empty.
        if BenchmarkRunner.isEnabled {
            print("=== PieVoice benchmark launch \(Date()) (console not mirrored) ===")
            return
        }
        let path = Self.path
        // Append across launches so a crash's last words survive the
        // relaunch — but don't let the file grow without bound. Rotate
        // rather than delete: the previous launch is exactly what a
        // post-mortem wants.
        if let size = try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int,
           size > 2 * 1024 * 1024 {
            let previous = path + ".prev"
            try? FileManager.default.removeItem(atPath: previous)
            try? FileManager.default.moveItem(atPath: path, toPath: previous)
        }
        freopen(path, "a", stdout)
        freopen(path, "a", stderr)
        setvbuf(stdout, nil, _IOLBF, 0)
        setvbuf(stderr, nil, _IONBF, 0)
        #endif
        print("=== PieVoice launch \(Date()) ===")
    }
}

/// One-line facts about the process that decide whether the engine can
/// run here, written to the console before the engine is touched.
///
/// The virtual-address ceiling is the number the collaborators asked
/// for: nobody publishes it for recent iPhones, and it is what sizes the
/// wasm pool and bounds the largest model that can be mapped.
enum LaunchDiagnostics {
    static func log() {
        let info = ProcessInfo.processInfo
        let physicalGB = Double(info.physicalMemory) / 1_073_741_824
        var machine = utsname()
        uname(&machine)
        let model = withUnsafePointer(to: &machine.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        print(String(
            format: "[launch] %@ iOS %@ · %.1f GB RAM · %d cores · va_ceiling≈%.1f GB",
            model,
            info.operatingSystemVersionString,
            physicalGB,
            info.activeProcessorCount,
            largestReservableGB()
        ))
        let selected = PieRuntimeConfig.selected
        print("[launch] model=\(selected.directory) artifact=\(selected.artifactPath ?? "missing")")
    }

    /// Largest single PROT_NONE reservation the kernel will grant, found
    /// by bisection. Reserving costs no physical memory and the mapping
    /// is released immediately, so this takes microseconds.
    static func largestReservableGB() -> Double {
        let gib: Int = 1 << 30
        func reservable(_ bytes: Int) -> Bool {
            let p = mmap(nil, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0)
            guard p != MAP_FAILED else { return false }
            munmap(p, bytes)
            return true
        }
        var low = 0
        var high = 1 << 40  // 1 TiB — comfortably beyond any iOS ceiling
        // Fast exit for hosts (the Simulator) that grant everything.
        if reservable(high) { return Double(high) / Double(gib) }
        while high - low > gib / 4 {
            let mid = low + (high - low) / 2
            if reservable(mid) { low = mid } else { high = mid }
        }
        return Double(low) / Double(gib)
    }
}

@main
struct PieVoiceApp: App {

    @StateObject private var composition = AppComposition()

    var body: some Scene {
        WindowGroup {
            ComposedRootView(composition: composition, settings: composition.settings)
        }
    }
}

/// Hands every view its collaborators and applies the app-wide look.
///
/// A view of its own so that it observes `AppSettings`: the scene body
/// only re-renders when the composition itself changes, which it never
/// does.
private struct ComposedRootView: View {
    let composition: AppComposition
    @ObservedObject var settings: AppSettings

    var body: some View {
        RootView()
            .environmentObject(composition.settings)
            .environmentObject(composition.store)
            .environmentObject(composition.chat)
            .environmentObject(composition.voice)
            .environmentObject(composition.dictation)
            .environmentObject(composition.router)
            .environmentObject(composition.voicePreview)
            .preferredColorScheme(settings.appearance.colorScheme)
            .tint(Theme.accent)
            .onChange(of: settings.appearance, initial: true) { _, appearance in
                WindowAppearance.apply(appearance)
            }
            .onAppear(perform: composition.launch)
    }
}

/// Sets the appearance on the app's windows directly.
///
/// `preferredColorScheme(nil)` only withdraws the request for a scheme;
/// the window can keep the one asked for last, so going from Dark back to
/// System would not follow the system until the next launch. Overriding
/// the window's style covers that, and reaches sheets and the voice-mode
/// cover too because they are presented in the same window.
private enum WindowAppearance {
    @MainActor
    static func apply(_ appearance: AppSettings.Appearance) {
        let style: UIUserInterfaceStyle
        switch appearance {
        case .system: style = .unspecified
        case .light: style = .light
        case .dark: style = .dark
        }
        for scene in UIApplication.shared.connectedScenes {
            guard let windowScene = scene as? UIWindowScene else { continue }
            for window in windowScene.windows {
                window.overrideUserInterfaceStyle = style
            }
        }
    }
}
