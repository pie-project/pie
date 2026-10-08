import SwiftUI

/// Composition root.
///
/// The only place where a concrete backend, a concrete microphone, and a
/// concrete synthesizer are named. Everything downstream sees protocols,
/// which is what keeps a Pie upgrade confined to `PieKit/` and an audio
/// change confined to `AudioKit/`.
final class AppComposition: ObservableObject {

    let controller: ConversationController
    /// Exposed so benchmark mode can drive the backend without the
    /// listen/speak loop in the way.
    let backend: ConversationBackend

    private let microphone = MicrophoneInput()
    private let sample = AudioFileInput(resources: AppComposition.sampleResources)

    /// A bundled utterance is optional — the app is fully usable without
    /// one, it just can't be driven hands-off.
    var hasSampleRecording: Bool { sample.url != nil }

    @Published var inputSource: InputSource = .microphone {
        didSet {
            guard oldValue != inputSource else { return }
            controller.useInput(inputSource == .microphone ? microphone : sample)
        }
    }

    /// Turns 2 and 3 are follow-ups on purpose: they are only coherent if
    /// the backend kept the conversation, so pressing "Sample" three
    /// times exercises the KV-snapshot path without anyone speaking.
    static let sampleResources = [
        "sample-question-1",
        "sample-question-2",
        "sample-question-3",
    ]

    init() {
        // Restore the ladder rung chosen on a previous launch before
        // anything reads the engine description or logs the model.
        PieRuntimeConfig.restoreSelection()
        ConsoleMirror.install()
        LaunchDiagnostics.log()

        let engine = PieEngine()
        backend = engine
        controller = ConversationController(
            backend: engine,
            input: microphone,
            output: SpokenOutput()
        )
    }
}

/// Mirrors stdout/stderr into a file inside the app container.
///
/// On a physical device there is no attached console, and devicectl's
/// console capture has proven unreliable — so everything the engine
/// prints (ggml load markers, wasmtime allocator choice, Rust panics)
/// goes to `Documents/pie-console.log`, which can be pulled off the
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
        print("[launch] model=\(PieRuntimeConfig.selected.fileName) present=\(PieRuntimeConfig.selected.isPresent)")
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
            VoiceChatView(
                controller: composition.controller,
                inputSource: Binding(
                    get: { composition.inputSource },
                    set: { composition.inputSource = $0 }
                ),
                hasSampleRecording: composition.hasSampleRecording
            )
            .onAppear {
                    if BenchmarkRunner.isEnabled {
                        // Benchmark mode drives the backend directly; the
                        // normal warm-up/listen path would race it.
                        BenchmarkRunner.run(backend: composition.backend)
                    } else {
                        composition.controller.bootstrap()
                    }
                }
        }
    }
}
