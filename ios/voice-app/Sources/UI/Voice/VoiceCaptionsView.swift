import SwiftUI

/// Voice mode's live captions: the user's words in gray, the reply in ink,
/// in the order they were said, pinned to the bottom of a fixed slot.
///
/// The slot never changes height, so a caption growing or clearing never
/// moves the orb. Lines that outgrow it slide up under a soft top edge, as
/// subtitles do, and new words fade in where they land while the words
/// before them stay put.
///
/// It watches only `VoiceCaptions`, which changes with every word; the
/// rest of voice mode does not redraw for them.
struct VoiceCaptionsView: View {
    @ObservedObject var captions: VoiceCaptions
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private struct Line: Identifiable {
        let id: String
        let text: String
        let color: Color
    }

    /// The non-empty captions, oldest first.
    private var lines: [Line] {
        let user = Line(id: "user", text: captions.user, color: Theme.secondaryInk)
        let reply = Line(id: "reply", text: captions.assistant, color: Theme.ink)
        let ordered = captions.replyComesFirst ? [reply, user] : [user, reply]
        return ordered.filter { !$0.text.isEmpty }
    }

    var body: some View {
        let lines = lines
        VStack(alignment: .leading, spacing: 14) {
            ForEach(lines) { line in
                FadingText(text: line.text, color: line.color)
                    .transition(.opacity)
            }
        }
        .font(.title3)
        .frame(maxWidth: .infinity, alignment: .leading)
        // As tall as the words need, even taller than the slot: the slot
        // below shows the bottom of it.
        .fixedSize(horizontal: false, vertical: true)
        // A caption appearing or clearing fades; lines pushed up by new
        // ones glide rather than jump a line at a time (with Reduce Motion
        // they move at once). Scoped to this small view, so it is cheap to
        // run per word.
        .animation(Motion.reduced(Motion.content, reduceMotion), value: lines.map(\.id))
        .animation(reduceMotion ? nil : Motion.content, value: captions.user)
        .animation(reduceMotion ? nil : Motion.content, value: captions.assistant)
        .padding(.horizontal, 24)
        .frame(maxHeight: .infinity, alignment: .bottom)
        .clipped()
        .mask(
            LinearGradient(
                stops: [
                    .init(color: .clear, location: 0),
                    .init(color: .black, location: 0.22),
                    .init(color: .black, location: 1),
                ],
                startPoint: .top,
                endPoint: .bottom
            )
        )
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.updatesFrequently)
    }
}

/// Text that fades in whatever is added to it, a word or a few at a time
/// as they arrive, while what was already there stays exactly where it is.
///
/// It is one `Text` made of runs: the settled part, then each recent
/// arrival at its own opacity. A run of the same string in the same font
/// lays out the same whatever its color, so nothing reflows as the runs
/// brighten. When the recogniser revises its last words, the revised
/// words fade in again from where the old and new text part ways.
struct FadingText: View {
    let text: String
    let color: Color

    @State private var reveal = RevealClock()
    /// The timeline runs only while some words are still fading. True at
    /// first, for the text the view appears with.
    @State private var isFading = true

    var body: some View {
        TimelineView(.animation(paused: !isFading)) { _ in
            // The wall clock, not the timeline's date: a paused timeline
            // keeps the date it stopped at.
            reveal.text(for: text, color: color, at: Date.timeIntervalSinceReferenceDate)
                // The runs are this view's own fade; SwiftUI's crossfade
                // of the whole string on top would dim the settled words.
                .contentTransition(.identity)
        }
        .onChange(of: text) { _, _ in isFading = true }
        .task(id: text) {
            // Once the newest words have finished fading, the timeline
            // stops until the text changes again.
            try? await Task.sleep(for: .seconds(RevealClock.fadeDuration + 0.05))
            if !Task.isCancelled { isFading = false }
        }
    }
}

/// When each part of a `FadingText` arrived. A plain class kept in
/// `@State`, updated as the text is drawn, so recording an arrival never
/// triggers another redraw.
@MainActor
final class RevealClock {
    /// How long one arrival takes to fade in: a little longer than
    /// `Motion.fadeIn`, so words that arrive close together overlap into a
    /// soft leading edge.
    static let fadeDuration: TimeInterval = 0.35

    private struct Arrival {
        /// Where it starts in the text, in characters.
        var start: Int
        var time: TimeInterval
    }

    /// The text the arrivals refer to.
    private var shown = ""
    /// Arrivals still fading, oldest first; everything before the first
    /// one has settled.
    private var arrivals: [Arrival] = []

    /// `text` as one `Text`, its recent arrivals at their current opacity.
    func text(for text: String, color: Color, at now: TimeInterval) -> Text {
        record(text, at: now)
        // Arrivals that have finished fading join the settled part.
        while let first = arrivals.first, now - first.time >= Self.fadeDuration {
            arrivals.removeFirst()
        }

        let characters = Array(text)
        let settledEnd = arrivals.first?.start ?? characters.count
        var result = Text(String(characters[..<settledEnd])).foregroundStyle(color)
        for (index, arrival) in arrivals.enumerated() {
            let end = index + 1 < arrivals.count ? arrivals[index + 1].start : characters.count
            let progress = min(max((now - arrival.time) / Self.fadeDuration, 0), 1)
            // Ease out: quick to show, slow to settle.
            let opacity = 1 - (1 - progress) * (1 - progress)
            result = result + Text(String(characters[arrival.start..<end])).foregroundStyle(color.opacity(opacity))
        }
        return result
    }

    /// Notes what changed since the text was last drawn: everything after
    /// the part the old and new text share is a new arrival.
    private func record(_ text: String, at now: TimeInterval) {
        guard text != shown else { return }
        let shared = zip(shown, text).prefix { $0 == $1 }.count
        shown = text
        // Arrivals in the part that changed are replaced by the new one.
        arrivals.removeAll { $0.start >= shared }
        if text.count > shared {
            arrivals.append(Arrival(start: shared, time: now))
        }
    }
}
