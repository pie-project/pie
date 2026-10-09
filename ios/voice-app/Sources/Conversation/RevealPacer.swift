import Foundation
import QuartzCore

/// Meters a streaming reply onto the screen a word at a time.
///
/// The engine's text arrives in bursts: a few tokens at once, then a
/// pause. Shown as it arrives, the reply jerks forward. ChatGPT instead
/// reveals words at a steady pace and speeds up only when it falls
/// behind, and this does the same: the received text waits in `pending`,
/// and once per screen refresh as many whole words move to `shown` as the
/// current rate allows.
///
/// The rate is `baseRate` while the reply keeps up, and rises so the
/// screen never trails the engine by more than `maxLag`. When the engine
/// is done, `finish()` shows the rest within `flushTime`. A word whose end
/// has not been received yet is held back (so `**bo` never flashes before
/// `**bold**`), unless nothing else has been shown for `partialWordWait`.
///
/// The unit is a word, as ChatGPT's is, except where words are not
/// separated by spaces. Chinese and Japanese (and Thai and its
/// neighbours) are shown a character at a time: taken as one "word" up to
/// the next space, a paragraph of them waited for `partialWordWait` and
/// then appeared at once, in jerks four times a second. A long run with
/// no space in Latin script (a URL, an identifier) is shown as it arrives
/// once it is `longWord` characters long, or once part of it is already
/// on screen, for the same reason.
///
/// This is for the screen only: voice mode speaks the engine's text as it
/// arrives and never goes through a pacer.
@MainActor
final class RevealPacer {
    /// Characters per second while the reply keeps up with the screen.
    private static let baseRate: Double = 200
    /// The screen never trails the engine by more than this.
    private static let maxLag: TimeInterval = 0.35
    /// After the engine is done, what is left is shown within this.
    private static let flushTime: TimeInterval = 0.25
    /// How long a half-received word may hold everything up.
    private static let partialWordWait: TimeInterval = 0.25
    /// A run without spaces this long is not waited for (see above).
    private static let longWord = 24

    /// What is on screen.
    private(set) var shown = ""
    /// Received, not yet on screen.
    private var pending = ""

    /// Called with all of `shown` each time more of it is revealed; at
    /// most once per screen refresh.
    private let onReveal: (String) -> Void

    private var displayLink: CADisplayLink?
    private var lastFrame: CFTimeInterval?
    private var lastReveal: CFTimeInterval = 0
    /// Characters the pace has earned but not spent yet. Goes negative
    /// when a long word is shown at once, which delays the next one.
    private var credit: Double = 0
    /// Set by `finish()`: the time by which everything must be shown.
    private var flushDeadline: CFTimeInterval?
    private var finishWaiter: CheckedContinuation<Void, Never>?

    init(onReveal: @escaping (String) -> Void) {
        self.onReveal = onReveal
    }

    /// Whether any text has been received at all.
    var hasReceived: Bool { !shown.isEmpty || !pending.isEmpty }

    /// Everything received so far, shown or not.
    var received: String { shown + pending }

    /// How long ago words were last revealed (0 if never).
    var timeSinceLastReveal: TimeInterval {
        lastReveal == 0 ? 0 : CACurrentMediaTime() - lastReveal
    }

    func receive(_ chunk: String) {
        guard !chunk.isEmpty else { return }
        pending += chunk
        startTicking()
    }

    /// The engine's final text can run a token past what was streamed;
    /// takes in whatever of `text` has not been received yet. A final text
    /// that does not continue the streamed one is left to the caller.
    func catchUp(to text: String) {
        let soFar = received
        guard text.count > soFar.count, text.hasPrefix(soFar) else { return }
        receive(String(text.dropFirst(soFar.count)))
    }

    /// Shows everything received so far at once (reasoning, when the
    /// reply proper starts).
    func flush() {
        guard !pending.isEmpty else { return }
        shown += pending
        pending = ""
        lastReveal = CACurrentMediaTime()
        onReveal(shown)
    }

    /// The engine is done: shows the rest quickly, then returns. Returns
    /// at once if nothing is waiting, and early if `stop()` is called.
    ///
    /// The display link does not run while the app is in the background,
    /// so a timer makes sure this returns anyway, with everything shown:
    /// the reply then settles and is saved as it would have been.
    func finish() async {
        guard !pending.isEmpty else { return }
        flushDeadline = CACurrentMediaTime() + Self.flushTime
        startTicking()
        let fallback = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64((Self.flushTime + 0.25) * 1_000_000_000))
            guard !Task.isCancelled, let self, self.finishWaiter != nil else { return }
            self.flush()
            self.stopTicking()
            self.resumeFinishWaiter()
        }
        await withCheckedContinuation { finishWaiter = $0 }
        fallback.cancel()
    }

    /// Freezes the screen where it is: what is shown stays, the rest is
    /// dropped, and no more reveals (or their haptics) happen.
    func stop() {
        pending = ""
        stopTicking()
        resumeFinishWaiter()
    }

    // MARK: - Frames

    private func startTicking() {
        guard displayLink == nil else { return }
        let link = CADisplayLink(target: FrameTarget(self), selector: #selector(FrameTarget.frame(_:)))
        // Word timing needs no more than 60 Hz; on a 120 Hz phone this
        // halves the main-thread wake-ups while the model decodes.
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        link.add(to: .main, forMode: .common)
        displayLink = link
        lastFrame = nil
        if lastReveal == 0 { lastReveal = CACurrentMediaTime() }
    }

    private func stopTicking() {
        displayLink?.invalidate()
        displayLink = nil
        lastFrame = nil
        credit = 0
    }

    private func resumeFinishWaiter() {
        flushDeadline = nil
        finishWaiter?.resume()
        finishWaiter = nil
    }

    fileprivate func frame(at now: CFTimeInterval) {
        let elapsed = lastFrame.map { now - $0 } ?? (1.0 / 60)
        lastFrame = now

        let backlog = Double(pending.count)
        var rate = max(Self.baseRate, backlog / Self.maxLag)
        if let deadline = flushDeadline {
            rate = max(rate, backlog / max(deadline - now, 1.0 / 120))
        }
        // A pause in the stream must not bank a burst for later.
        credit = min(credit + rate * elapsed, max(1, rate / 30))

        let holdingTooLong = now - lastReveal > Self.partialWordWait
        var revealed = false
        while credit > 0, let unit = nextWord(includingPartial: flushDeadline != nil || holdingTooLong) {
            shown += unit
            pending.removeFirst(unit.count)
            credit -= Double(unit.count)
            revealed = true
        }
        if revealed {
            lastReveal = now
            onReveal(shown)
        }
        if pending.isEmpty {
            stopTicking()
            resumeFinishWaiter()
        }
    }

    /// The next thing to show: the whitespace before the next word plus
    /// the word itself, so a paragraph break and the word after it appear
    /// together. In a script written without spaces the "word" is one
    /// character. Nil if that word is still arriving and may not be shown
    /// partly yet.
    private func nextWord(includingPartial: Bool) -> String? {
        guard !pending.isEmpty else { return nil }
        var end = pending.startIndex
        while end < pending.endIndex, pending[end].isWhitespace { end = pending.index(after: end) }
        guard end < pending.endIndex else {
            // Whitespace with no word after it yet: wait for the word.
            return includingPartial ? pending : nil
        }
        if pending[end].isWrittenWithoutSpaces {
            // A whole character: shown on its own.
            return String(pending[...end])
        }
        let wordStart = end
        while end < pending.endIndex, !pending[end].isWhitespace, !pending[end].isWrittenWithoutSpaces {
            end = pending.index(after: end)
        }
        if end == pending.endIndex, !includingPartial {
            // The word may still be arriving. Wait for its end, unless it
            // is long, or it carries on a word already partly on screen
            // (nothing here to hold back that is not already showing).
            let continuesShownWord = wordStart == pending.startIndex
                && shown.last.map { !$0.isWhitespace } ?? false
            let isLong = pending.distance(from: wordStart, to: end) >= Self.longWord
            guard continuesShownWord || isLong else { return nil }
        }
        return String(pending[..<end])
    }
}

private extension Character {
    /// A character of a script whose words are not separated by spaces:
    /// Chinese characters, Japanese kana, their punctuation and full-width
    /// forms, and Thai, Lao, Khmer and Myanmar. Korean separates words
    /// with spaces and is not included.
    var isWrittenWithoutSpaces: Bool {
        guard let scalar = unicodeScalars.first, scalar.value >= 0x0E00 else { return false }
        switch scalar.value {
        case 0x0E00...0x0EFF,   // Thai, Lao
             0x1000...0x109F,   // Myanmar
             0x1780...0x17FF,   // Khmer
             0x2E80...0x2FDF,   // CJK radicals
             0x3000...0x30FF,   // CJK punctuation, Hiragana, Katakana
             0x3400...0x4DBF,   // CJK extension A
             0x4E00...0x9FFF,   // CJK unified ideographs
             0xF900...0xFAFF,   // CJK compatibility ideographs
             0xFF00...0xFFEF,   // full-width forms and punctuation
             0x20000...0x3FFFF: // CJK extensions B and later
            return true
        default:
            return false
        }
    }
}

/// The display link's target. A display link keeps its target alive, so
/// this holds the pacer weakly and lets it go away with its reply.
private final class FrameTarget: NSObject {
    weak var pacer: RevealPacer?

    init(_ pacer: RevealPacer) {
        self.pacer = pacer
    }

    @objc func frame(_ link: CADisplayLink) {
        MainActor.assumeIsolated {
            guard let pacer else {
                link.invalidate()
                return
            }
            pacer.frame(at: link.timestamp)
        }
    }
}
