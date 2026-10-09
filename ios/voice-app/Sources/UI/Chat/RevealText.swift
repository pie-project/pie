import SwiftUI

/// A piece of a reply's text that fades its newest words in while the
/// reply streams: ChatGPT's soft leading edge.
///
/// The pacer reveals a word or two at a time. Each time the text gets
/// longer, this remembers where it ended and when (`marks`). The text is
/// drawn as one `Text`, cut into segments at those marks, each tagged
/// with the time it appeared, and a `TextRenderer` draws every segment at
/// an opacity that grows from 0 to 1 over `fadeDuration` after its time.
/// So the last two to four words are always part way in, and text already
/// on screen never moves: only opacity changes.
///
/// A `TimelineView` redraws the renderer while a fade is in progress, at
/// most 60 times a second (a quarter-second opacity ramp looks the same
/// at 120, and on a ProMotion phone the model is decoding on the same
/// chip), and pauses once the newest words are fully in. The text and its
/// layout are built once per reveal, not per frame.
///
/// Before iOS 18 (no `TextRenderer`) and for finished replies this is a
/// plain `Text`; the pacer still shows the words at a steady pace.
///
/// The paragraph that is growing ends in a few invisible words
/// (`padsLastLine`). Text lays out with "push-out" line breaking: a word
/// that would sit alone on a paragraph's last line takes the word before
/// it down with it. While the reply streams, every new word is for a
/// moment the paragraph's last word, so a word already on screen jumped
/// down a line and back up a frame later (recorded at most line wraps).
/// With the invisible words after it, the real words are never the last
/// line's only word, so they stay where they were first drawn.
struct RevealText: View {
    let text: AttributedString
    /// Part of a reply that is still streaming.
    let isLive: Bool
    /// End the text with invisible words while it is live; see above. For
    /// the paragraph a streaming reply is writing into, not for code
    /// (which does not wrap) or paragraphs that are finished.
    var padsLastLine = false

    /// How long a newly revealed word takes to fade in.
    static let fadeDuration: TimeInterval = 0.25
    /// The fade's redraw interval: 60 Hz is plenty for it (see above).
    private static let frameInterval: TimeInterval = 1.0 / 60

    /// Where the text ended at each reveal, and when; ascending.
    @State private var marks: [RevealMark]
    /// The newest words are fully in, so the timeline can pause.
    @State private var isSettled = true

    /// - Parameter startsFresh: the text appeared just now with this view
    ///   (a new paragraph of a streaming reply), so all of it fades in.
    ///   Otherwise what it holds when it first appears is already shown.
    init(_ text: AttributedString, isLive: Bool, startsFresh: Bool, padsLastLine: Bool = false) {
        self.text = text
        self.isLive = isLive
        self.padsLastLine = padsLastLine
        let shown = startsFresh ? [] : [RevealMark(end: text.characters.count, time: 0)]
        _marks = State(initialValue: shown)
    }

    var body: some View {
        if isLive, #available(iOS 18.0, *) {
            fading
                .transition(.identity)
        } else if isLive && padsLastLine {
            // iOS 17: no fade, but the same invisible words.
            Text("\(Text(text))\(Text(Self.padding).foregroundStyle(.clear))")
                .accessibilityLabel(Text(text))
                .transition(.identity)
        } else {
            Text(text)
                .transition(.identity)
        }
    }

    /// The invisible words at the end of a growing paragraph. Three, so
    /// that when push-out moves a word down, it is one of these.
    private static let padding = " mm mm mm"

    @available(iOS 18.0, *)
    private var fading: some View {
        let segmented = Self.segmented(text, at: marks, padded: padsLastLine)
        let drawnAt = Date().timeIntervalSinceReferenceDate
        return TimelineView(.animation(minimumInterval: Self.frameInterval, paused: isSettled)) { context in
            segmented.textRenderer(RevealFade(
                now: isSettled ? drawnAt : context.date.timeIntervalSinceReferenceDate
            ))
            // VoiceOver reads the text, not the invisible words.
            .accessibilityLabel(Text(text))
        }
        .onChange(of: text.characters.count, initial: true) { _, length in
            record(length)
        }
        .task(id: marks.last?.time) {
            try? await Task.sleep(nanoseconds: UInt64((Self.fadeDuration + 0.05) * 1_000_000_000))
            if !Task.isCancelled { isSettled = true }
        }
    }

    /// Notes that the text is now `length` characters long. Marks whose
    /// fade is over are merged into one, so there are only ever a few.
    private func record(_ length: Int) {
        let now = Date().timeIntervalSinceReferenceDate
        var kept: [RevealMark] = []
        if let settledEnd = marks.filter({ now - $0.time >= Self.fadeDuration }).map(\.end).max() {
            kept.append(RevealMark(end: settledEnd, time: 0))
        }
        kept += marks.filter { now - $0.time < Self.fadeDuration }
        // The text can also get shorter, when the Markdown of the last
        // words is read differently once more of it arrives.
        var next: [RevealMark] = []
        for mark in kept {
            let end = min(mark.end, length)
            if end > (next.last?.end ?? 0) { next.append(RevealMark(end: end, time: mark.time)) }
        }
        if length > (next.last?.end ?? 0) {
            next.append(RevealMark(end: length, time: now))
        }
        guard next != marks else { return }
        marks = next
        isSettled = false
    }

    /// `text` as one `Text` cut at the marks, each segment tagged with the
    /// time it appeared. Text past the last mark has not been recorded yet
    /// (it arrived this very update) and is tagged to stay hidden until it
    /// is, a moment later. With `padded`, the invisible words follow,
    /// tagged never to be drawn.
    ///
    /// The marks were recorded for the text as it was at the last update,
    /// and the text can since have got shorter: the Markdown of the last
    /// words is sometimes read differently once more of it arrives
    /// (`[Apple` becomes the link `Apple`). Each mark is clamped to the
    /// text's length here, because indexing past the end of the text
    /// crashes; `record` tidies the marks up a moment later.
    @available(iOS 18.0, *)
    private static func segmented(_ text: AttributedString, at marks: [RevealMark], padded: Bool) -> Text {
        let characters = text.characters
        let length = characters.count
        var pieces: [Text] = []
        var start = characters.startIndex
        var offset = 0
        for mark in marks + [RevealMark(end: length, time: .infinity)] {
            let end = min(mark.end, length)
            guard end > offset else { continue }
            let endIndex = characters.index(start, offsetBy: end - offset)
            let piece = Text(AttributedString(text[start..<endIndex]))
            pieces.append(mark.time == 0 ? piece : piece.customAttribute(RevealTime(time: mark.time)))
            start = endIndex
            offset = end
        }
        if padded {
            pieces.append(Text(padding).customAttribute(RevealTime(time: .infinity)))
        }
        return pieces.reduce(Text(verbatim: "")) { Text("\($0)\($1)") }
    }
}

/// The end of the text, in characters, as of one reveal. A `time` of 0
/// means shown long ago.
private struct RevealMark: Equatable {
    var end: Int
    var time: TimeInterval
}

/// When a segment of streaming text appeared.
private struct RevealTime: TextAttribute {
    var time: TimeInterval
}

/// Draws each segment of the text at the opacity its age gives it.
@available(iOS 18.0, *)
private struct RevealFade: TextRenderer {
    var now: TimeInterval

    func draw(layout: Text.Layout, in context: inout GraphicsContext) {
        for line in layout {
            for run in line {
                guard let reveal = run[RevealTime.self] else {
                    context.draw(run)
                    continue
                }
                let progress = (now - reveal.time) / RevealText.fadeDuration
                if progress >= 1 {
                    context.draw(run)
                } else if progress > 0 {
                    var faded = context
                    // Ease out: quick to become readable, gentle to finish.
                    faded.opacity = progress * (2 - progress)
                    faded.draw(run)
                }
            }
        }
    }
}
