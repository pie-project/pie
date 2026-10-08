import Foundation

/// Self-measuring benchmark pass, so device numbers are produced by the app
/// rather than read off screenshots.
///
/// Enabled with the `-PieBenchmark 1` launch argument; the normal UI never
/// sees it. Every result line is printed as `PIEBENCH {json}` so a driver
/// script can capture them from the console of either the Simulator or a
/// physical device.
///
/// What it measures, per turn:
///   - `ttft_s`      time from request to the first token reaching the app
///   - `decode_tps`  generated tokens ÷ time spent generating after TTFT
///   - `prefill`     tokens the engine had to prefill this turn
///   - `reused`      conversation tokens served from the KV snapshot
///   - `footprint_mb` physical memory footprint — the figure jetsam judges
enum BenchmarkRunner {

    static var isEnabled: Bool {
        UserDefaults.standard.string(forKey: "PieBenchmark") == "1"
    }

    /// Turns to run. Defaults to five: turn 1 is a fresh conversation, and
    /// 2–5 exercise the KV-reuse path as history grows.
    private static var turnCount: Int {
        let n = UserDefaults.standard.integer(forKey: "PieBenchmarkTurns")
        return n > 0 ? n : 5
    }

    private static let script = [
        "What is one good reason to run a language model on a phone instead of in the cloud?",
        "Can you say that more simply?",
        "What is the hardest part of doing that?",
        "Give me one example of where that would matter.",
        "How would you explain that to someone new?",
    ]

    /// Physical footprint in MB — what iOS actually counts against the app.
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return -1 }
        return Double(info.phys_footprint) / 1_048_576
    }

    /// Removes every model in Documents except the one this run wants.
    ///
    /// The ladder pushes one multi-gigabyte model per rung; without this a
    /// device accumulates all of them (~8 GB) in the app container. Gated
    /// behind an explicit flag so running benchmark mode by hand never
    /// deletes anything unexpectedly — only the driver passes it.
    private static func pruneOtherModels(keeping keepFile: String) {
        guard UserDefaults.standard.string(forKey: "PieBenchPruneModels") == "1" else { return }
        let fm = FileManager.default
        let dir = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("qwen3-gguf")
        guard let entries = try? fm.contentsOfDirectory(atPath: dir.path) else { return }
        for name in entries where name != keepFile && name.hasSuffix(".gguf") {
            try? fm.removeItem(at: dir.appendingPathComponent(name))
            emit(["event": "pruned_model", "file": name])
        }
    }

    private static func emit(_ object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let json = String(data: data, encoding: .utf8) else { return }
        print("PIEBENCH \(json)")
        fflush(stdout)
    }

    /// Runs the whole pass and terminates the process, so a driver script
    /// can wait on exit rather than guessing when it finished.
    static func run(backend: ConversationBackend) {
        Task.detached(priority: .userInitiated) {
            let device = await deviceDescription()
            if let requested = PieRuntimeConfig.requestedModelFile {
                pruneOtherModels(keeping: requested)
            }

            // A silently-substituted model would produce a table of numbers
            // attributed to the wrong weights. Refuse rather than mislead.
            if let requested = PieRuntimeConfig.requestedModelFile,
               requested != PieRuntimeConfig.selected.fileName {
                emit([
                    "event": "model_mismatch",
                    "requested": requested,
                    "loaded": PieRuntimeConfig.selected.fileName,
                    "available": PieRuntimeConfig.available.map(\.fileName),
                ])
                try? await Task.sleep(nanoseconds: 300_000_000)
                exit(2)
            }
            emit([
                "event": "start",
                "model": PieRuntimeConfig.modelDescription,
                "engine": backend.engineDescription,
                "device": device,
                "footprint_mb_baseline": footprintMB(),
            ])

            // Engine boot + weight load, paid once.
            let bootStart = Date()
            if let bootError = await backend.warmUp() {
                emit(["event": "boot_error", "error": bootError])
                try? await Task.sleep(nanoseconds: 300_000_000)
                exit(3)
            }
            emit([
                "event": "warmup",
                "seconds": -bootStart.timeIntervalSinceNow,
                "footprint_mb": footprintMB(),
            ])

            var peak = footprintMB()

            for index in 0..<turnCount {
                let utterance = script[index % script.count]
                let started = Date()
                var firstTokenAt: Date?

                do {
                    let result = try await backend.reply(
                        to: utterance,
                        startingFresh: index == 0
                    ) { _ in
                        if firstTokenAt == nil { firstTokenAt = Date() }
                    }

                    let ttft = (firstTokenAt ?? Date()).timeIntervalSince(started)
                    let total = -started.timeIntervalSinceNow
                    // Decode rate over the generation window only: including
                    // prefill would flatter or punish the number depending on
                    // prompt length rather than reporting decode speed.
                    let decodeWindow = max(total - ttft, 0.0001)
                    let footprint = footprintMB()
                    peak = max(peak, footprint)

                    emit([
                        "event": "turn",
                        "turn": index + 1,
                        "utterance": utterance,
                        "reused": result.stats.reused,
                        "prefill": result.stats.newPrefill,
                        "generated": result.stats.generated,
                        "ttft_s": ttft,
                        "total_s": total,
                        "decode_tps": Double(result.stats.generated) / decodeWindow,
                        "footprint_mb": footprint,
                        "chars": result.text.count,
                        "note": result.stats.note,
                    ])
                } catch {
                    emit([
                        "event": "turn_error",
                        "turn": index + 1,
                        "error": String(describing: error),
                        "footprint_mb": footprintMB(),
                    ])
                }
            }

            emit([
                "event": "done",
                "model": PieRuntimeConfig.modelDescription,
                "peak_footprint_mb": peak,
            ])
            // Give the console a beat to flush before the process dies.
            try? await Task.sleep(nanoseconds: 400_000_000)
            exit(0)
        }
    }

    @MainActor
    private static func deviceDescription() -> String {
        var info = utsname()
        uname(&info)
        let machine = withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        #if targetEnvironment(simulator)
        return "simulator:\(machine)"
        #else
        return "device:\(machine)"
        #endif
    }
}
