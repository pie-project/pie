import Foundation

/// Turns a token stream into speakable sentences.
///
/// Without this the app would wait for the whole reply before making a
/// sound, which on a phone-class model is several seconds of silence
/// after every question. Emitting on sentence boundaries means speech
/// starts once the first sentence lands and the rest generates while the
/// synthesizer is still talking.
final class SentenceChunker {

    /// Below this, a boundary is ignored — otherwise "Sure." and "Yes."
    /// each become their own utterance and the speech sounds chopped.
    private let minimumLength = 12
    /// Above this, emit at the last breathing space even without a
    /// terminator, so a model that forgets to punctuate is still audible.
    private let forcedLength = 180

    private var buffer = ""

    /// Feeds a delta and returns any sentences that are now complete.
    func push(_ delta: String) -> [String] {
        buffer += delta
        var sentences: [String] = []
        while let sentence = takeSentence() {
            sentences.append(sentence)
        }
        return sentences
    }

    /// Returns whatever is left at the end of a turn.
    func flush() -> String? {
        let remainder = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer = ""
        return remainder.isEmpty ? nil : remainder
    }

    func reset() {
        buffer = ""
    }

    // MARK: - Boundaries

    private func takeSentence() -> String? {
        if let index = boundaryIndex() {
            return take(upTo: index)
        }
        if buffer.count > forcedLength, let index = lastBreathingSpace() {
            return take(upTo: index)
        }
        return nil
    }

    /// A terminator only ends a sentence when whitespace already follows
    /// it in the buffer. That single rule keeps "3.14" and "Dr. Zhong"
    /// intact, and it naturally waits for more tokens instead of guessing.
    private func boundaryIndex() -> String.Index? {
        let terminators: Set<Character> = [".", "!", "?", "\n"]
        var index = buffer.startIndex
        var scanned = 0

        while index < buffer.endIndex {
            let character = buffer[index]
            let next = buffer.index(after: index)
            scanned += 1

            if terminators.contains(character), scanned >= minimumLength {
                if character == "\n" {
                    return next
                }
                if next < buffer.endIndex, buffer[next].isWhitespace {
                    return next
                }
            }
            index = next
        }
        return nil
    }

    private func lastBreathingSpace() -> String.Index? {
        let breaks: Set<Character> = [",", ";", ":", " "]
        var candidate: String.Index?
        var index = buffer.startIndex
        var scanned = 0

        while index < buffer.endIndex {
            if breaks.contains(buffer[index]), scanned >= minimumLength {
                candidate = buffer.index(after: index)
            }
            index = buffer.index(after: index)
            scanned += 1
        }
        return candidate
    }

    private func take(upTo index: String.Index) -> String? {
        let sentence = String(buffer[buffer.startIndex..<index])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        buffer = String(buffer[index...])
        return sentence.isEmpty ? nil : sentence
    }
}
