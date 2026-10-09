import Foundation

/// Tells the user's words apart from a reply's own words heard back.
///
/// The echo canceller takes most of a reply out of the microphone, but not
/// all of it, and now and then what is left is loud enough to start an
/// utterance and be transcribed. What leaks is the reply itself, in the
/// reply's own order, heard imperfectly: the recogniser drops a word here,
/// hears "personnel" for "personal" or "your" for "any" there, and turns a
/// name into words it knows ("Lin Jong" for "Lynn Zhang"). So the filter
/// follows the reply's word sequences through the transcript, accepting at
/// each step a word that sounds like the one the reply said there, and asks
/// whether what is left over is someone saying something else.
///
/// While the reply plays it leans towards echo. The reply mistaken for the
/// user is put to the model as the next question, whose answer can leak in
/// turn; the user mistaken for the reply only has to say it again, or
/// start with "wait". Once the reply has stopped, only the words it was
/// saying as it stopped can reach an utterance, through the detector's
/// pre-roll, and only at its start, so whatever follows them is the user's.
///
/// Foundation only, so it can be exercised off the device.
struct EchoFilter {

    enum Verdict: Equatable {
        /// The user. The text is the transcript without the reply's words
        /// leading it: the pre-roll and the recogniser often start the
        /// user's utterance with the end of what the reply was saying.
        case user(String)
        /// The reply heard back.
        case echo
        /// Too little to tell yet, such as one or two words the reply
        /// could have said.
        case undecided
    }

    /// When the utterance began, relative to the reply.
    enum Timing: Equatable {
        /// While the reply was playing: any of it can be in the transcript.
        case during
        /// Just after the reply played to its end: only its last words can
        /// be, and only at the start of the transcript.
        case afterEnd
        /// Just after the reply was stopped part way: only the words it was
        /// saying as it stopped can be, at the start of the transcript.
        /// Where it stopped is not known, so any of it is considered.
        case afterStop
    }

    private let reply: [Word]
    private let timing: Timing
    /// Where each reply word occurs, by `Word.key`.
    private let positions: [String: [Int]]
    /// The reply's distinctive words, one of each key: the only ones
    /// matched loosely wherever they are.
    private let distinctive: [Word]
    /// The reply's common words long enough for a word one letter off to
    /// be one of them misheard ("migt" for "might"), one of each.
    private let longCommon: [[Character]]
    /// The groups of `homophones` the reply has a word from.
    private let homophonesSaid: Set<Int>

    /// `reply` is everything the transcript could be an echo of: what was
    /// handed to the synthesizer, in the form it was handed.
    init(reply: String, timing: Timing = .during) {
        var words = Self.words(in: reply)
        if timing == .afterEnd {
            words = Array(words.suffix(Self.endingWords))
        }
        self.reply = words
        self.timing = timing
        var positions: [String: [Int]] = [:]
        var distinctive: [String: Word] = [:]
        var longCommon: Set<String> = []
        for (index, word) in words.enumerated() {
            positions[word.key, default: []].append(index)
            if !word.isCommon {
                distinctive[word.key] = word
            } else if word.form.count >= 4 {
                longCommon.insert(word.form)
            }
        }
        self.positions = positions
        self.distinctive = Array(distinctive.values)
        self.longCommon = longCommon.map(Array.init)
        homophonesSaid = Set(words.compactMap(\.homophoneGroup))
    }

    func judge(_ transcript: String) -> Verdict {
        let heard = Alignment(transcript, against: self)
        guard !heard.words.isEmpty else { return .undecided }
        switch timing {
        case .during:
            return judgeOverlapping(heard)
        case .afterEnd, .afterStop:
            return judgeAfterReply(heard)
        }
    }

    /// The transcript opens with a request to stop ("stop", "wait",
    /// "hold on", perhaps after an "oh" or an "okay") that is not the
    /// reply's own word heard back. Safe to act on before the utterance is
    /// over: nothing the user goes on to say changes what they asked for.
    func asksToStop(_ transcript: String) -> Bool {
        let heard = Alignment(transcript, against: self)
        return heard.interruption(at: 0)?.asksToStop == true
    }

    /// An utterance that overlapped the reply.
    private func judgeOverlapping(_ heard: Alignment) -> Verdict {
        let words = heard.words
        if words.count < Self.wordsToDecide {
            if let interrupting = heard.interruptionAfterReply {
                return .user(heard.text(from: interrupting))
            }
            // A question word or two the reply never said: "why?",
            // "what's glucose?".
            if Self.questionWords.contains(words[0].form), heard.matches[0].isEmpty {
                return .user(heard.text(from: 0))
            }
            // Two content words the reply never said, not even loosely: a
            // short question ("Berlin instead"). One word could be a
            // misheard word of the reply's, and waits for more.
            if words.count == 2, words.indices.allSatisfy(heard.isForeignContent) {
                return .user(heard.text(from: 0))
            }
            return .undecided
        }

        if let interrupting = heard.interruptionAfterReply {
            return .user(heard.text(from: interrupting))
        }
        // A question the reply did not say: what comes before it that the
        // reply did not say is the user's too.
        if let question = heard.words.indices.first(where: heard.opensQuestion) {
            return .user(heard.text(from: min(question, heard.userStart)))
        }
        let start = heard.userStart
        guard start < words.count else { return .echo }
        if heard.goesPastReply {
            return .user(heard.text(from: start))
        }
        let rest = start..<words.count
        let foreignContent = rest.filter(heard.isForeignContent).count
        guard heard.resemblesReply else {
            // Nothing in it follows the reply. Words that are all
            // function words ("I'm in a") say nothing either way.
            return foreignContent > 0 ? .user(heard.text(from: start)) : .undecided
        }
        // Some of it follows the reply, so the rest has to be clearly
        // somebody else's: a stretch of new words that says something
        // ("what about Berlin"), or new content words in a transcript that
        // is not mostly the reply. A misheard echo leaves isolated
        // strangers ("list" for "experiences") rather than stretches.
        if heard.longestForeignStretch(in: rest) >= Self.foreignStretch
            || (foreignContent >= 2 && heard.explainedShare < Self.echoShare) {
            return .user(heard.text(from: start))
        }
        return .echo
    }

    /// An utterance that began once the reply had stopped: the reply's
    /// words can only lead it, and whatever follows them is the user's.
    private func judgeAfterReply(_ heard: Alignment) -> Verdict {
        let words = heard.words
        let start = timing == .afterEnd ? heard.endingLength : heard.leadingRunLength
        guard start < words.count else { return .echo }
        // One word the reply said, on its own, could be either.
        if words.count == 1, !heard.matches[0].isEmpty {
            return .undecided
        }
        // "um", "the": nothing to answer.
        guard words[start...].contains(where: { !$0.isFunctionWord || Self.answers.contains($0.form) }) else {
            return .undecided
        }
        return .user(heard.text(from: start))
    }

    /// Fewer words than this that the reply could have said are left
    /// undecided.
    private static let wordsToDecide = 3
    /// The share of words the reply accounts for that makes a transcript
    /// the reply's: of its content words, for whether it resembles the
    /// reply at all; of all its words, for whether new content words in it
    /// are still not enough to make it the user's.
    private static let echoShare = 0.6
    /// Consecutive new words that say something (two content words, or a
    /// question) and make the user however much of the reply surrounds
    /// them.
    private static let foreignStretch = 3
    /// How many of a reply's last words an utterance that began after it
    /// finished is set against. The detector's pre-roll reaches 1 s back,
    /// two or three words of a reply; the margin allows for a slow voice.
    private static let endingWords = 8
    /// How far short of the reply's last word the words leading an
    /// utterance that began after it may stop and still be its ending: the
    /// recogniser often loses the last word or two of what it heard.
    private static let endingSlack = 2
    /// The least spelling similarity at which two distinctive words of four
    /// letters or more count as the same word misheard wherever they are:
    /// one letter in four.
    private static let misheardSimilarity = 0.75
    /// The least spelling similarity at which a word sounds like the
    /// reply's word due at its place.
    private static let nearSimilarity = 0.6

    // MARK: - Alignment

    /// A stretch of the transcript that follows the reply word for word,
    /// give or take words misheard, missed or added.
    private struct Run {
        let start: Int
        /// Inclusive; a word in step with the reply.
        let end: Int
        /// Words in the stretch that match a reply word outright, rather
        /// than only sounding like the one due.
        let matched: Int
        /// Words in the stretch in step with the reply, matched or
        /// sounding like the word due.
        let steps: Int
        /// The reply positions `start` and `end` stand for.
        let replyStart: Int
        let replyEnd: Int
    }

    /// One transcript set against the reply.
    private struct Alignment {
        let filter: EchoFilter
        let transcript: String
        let words: [Word]
        /// The reply positions each word matches outright.
        let matches: [Set<Int>]
        /// Sounds like a distinctive word the reply says somewhere, closely
        /// enough to count wherever it is.
        let soundsLikeReply: [Bool]
        /// Non-overlapping, in order, each with two words that match the
        /// reply outright, or one and two more in step.
        private(set) var runs: [Run] = []
        /// The run each word is in, by its index in `runs`.
        private(set) var runOf: [Int?] = []
        private(set) var inRun: [Bool] = []

        init(_ transcript: String, against filter: EchoFilter) {
            self.filter = filter
            self.transcript = transcript
            let words = EchoFilter.words(in: transcript)
            self.words = words
            matches = words.map(filter.positions(matching:))
            soundsLikeReply = words.indices.map { index in
                filter.isLooselyInReply(words[index], next: index + 1 < words.count ? words[index + 1] : nil,
                                        previous: index > 0 ? words[index - 1] : nil)
            }

            var found: [Run] = []
            var index = 0
            while index < words.count {
                let free = found.last.map { $0.end + 1 } ?? 0
                let best = matches[index]
                    .map { extendedBack(follow(from: index, at: $0), notBefore: free) }
                    .max { ($0.matched, $0.steps, $0.end) < ($1.matched, $1.steps, $1.end) }
                guard let best, best.matched >= 2 || best.steps >= 3 else {
                    index += 1
                    continue
                }
                found.append(best)
                index = best.end + 1
            }
            runs = mergedAcrossGaps(found)
            runOf = Array(repeating: nil, count: words.count)
            for (number, run) in runs.enumerated() {
                for covered in run.start...run.end { runOf[covered] = number }
            }
            inRun = runOf.map { $0 != nil }
            userStart = findUserStart()
        }

        /// Two runs with as many words between them as the reply has
        /// between the words they stand for, give or take one, are one
        /// stretch of the reply with those words misheard ("glucose and
        /// oxygen i hear market suggest in the leaves"), unless the words
        /// between ask or answer something: the user talking in the middle
        /// of the reply's echo.
        private func mergedAcrossGaps(_ runs: [Run]) -> [Run] {
            var merged: [Run] = []
            for run in runs {
                if let previous = merged.last {
                    let between = (previous.end + 1)..<run.start
                    let replyBetween = run.replyStart - previous.replyEnd - 1
                    if replyBetween >= 0, between.count <= Gap.longest, abs(between.count - replyBetween) <= 1,
                       !between.contains(where: isAnswer) {
                        merged[merged.count - 1] = Run(
                            start: previous.start, end: run.end, matched: previous.matched + run.matched,
                            steps: previous.steps + run.steps, replyStart: previous.replyStart, replyEnd: run.replyEnd
                        )
                        continue
                    }
                }
                merged.append(run)
            }
            return merged
        }

        // MARK: Following the reply

        /// Matches the reply's word at `position` outright.
        private func matches(_ index: Int, _ position: Int) -> Bool {
            index < words.count && matches[index].contains(position)
        }

        /// Sounds like the reply's word at `position`, the word due there.
        private func soundsLike(_ index: Int, _ position: Int) -> Bool {
            index < words.count && position >= 0 && position < filter.reply.count
                && EchoFilter.soundsAlike(words[index], filter.reply[position])
        }

        private func inStep(_ index: Int, _ position: Int) -> Bool {
            matches(index, position) || soundsLike(index, position)
        }

        /// Words `index` and `index + 1` are the reply's word at
        /// `position` heard as two ("photo synthesis", "to day").
        private func splits(_ index: Int, _ position: Int) -> Bool {
            guard index + 1 < words.count, position >= 0, position < filter.reply.count else { return false }
            return EchoFilter.isSimilar(words[index].letters + words[index + 1].letters, filter.reply[position].letters,
                                        atLeast: EchoFilter.misheardSimilarity)
        }

        /// The reply picks up again at word `index`, reply word
        /// `position`, after `gap`, a stretch the recogniser got wrong. On a
        /// word that only sounds like the reply's it needs the word after in
        /// step too, and so does a longer stretch, unless the reply picks
        /// up on a distinctive word of its own after as many words as it
        /// said there, or ends the transcript on one after words that
        /// neither ask nor answer anything.
        private func resumes(_ index: Int, _ position: Int, after gap: Gap) -> Bool {
            let confirmed = inStep(index + 1, position + 1)
            guard matches(index, position) else {
                return soundsLike(index, position) && confirmed
            }
            if gap.isSmall || confirmed { return true }
            guard !words[index].isCommon else { return false }
            return gap.words == gap.reply
                || (index == words.count - 1 && !((index - gap.words)..<index).contains(where: isAnswer))
        }

        /// Follows the reply from transcript word `start`, matched to
        /// reply word `position`, for as long as it keeps step. Words the
        /// recogniser got wrong are forgiven where the reply picks up again
        /// right after them, and words that sound like the reply's word due
        /// keep it in step.
        func follow(from start: Int, at position: Int) -> Run {
            var word = start
            var replyWord = position
            var matched = 0
            var steps = 0
            var end = start
            var replyEnd = position
            while word < words.count, replyWord < filter.reply.count {
                if matches(word, replyWord) {
                    matched += 1
                    steps += 1
                    end = word
                    replyEnd = replyWord
                    word += 1
                    replyWord += 1
                } else if soundsLike(word, replyWord) {
                    steps += 1
                    end = word
                    replyEnd = replyWord
                    word += 1
                    replyWord += 1
                } else if splits(word, replyWord) {
                    steps += 1
                    end = word + 1
                    replyEnd = replyWord
                    word += 2
                    replyWord += 1
                } else if let gap = Gap.all.first(where: { resumes(word + $0.words, replyWord + $0.reply, after: $0) }) {
                    word += gap.words
                    replyWord += gap.reply
                } else {
                    break
                }
            }
            return Run(start: start, end: end, matched: matched, steps: steps, replyStart: position, replyEnd: replyEnd)
        }

        /// What the recogniser can get wrong between two words in step: so
        /// many words heard in place of so many of the reply's.
        struct Gap {
            let words: Int
            let reply: Int

            /// One word misheard, missed or added.
            var isSmall: Bool { words <= 1 && reply <= 1 }

            /// Smallest first. (1, 1) is a word misheard, (0, 1) a reply
            /// word missed, (1, 0) a word added, (2, 1) one word heard as
            /// two.
            static let all = [(1, 1), (0, 1), (1, 0), (2, 2), (2, 1), (1, 2), (0, 2), (3, 3)]
                .map { Gap(words: $0.0, reply: $0.1) }

            /// The most words heard wrong in a row that can still be the
            /// reply's.
            static let longest = 4
        }

        /// `run`, taking in the words before it that sound like the reply's
        /// words due there: a name misheard as it began ("Lin Jong is
        /// doing").
        private func extendedBack(_ run: Run, notBefore free: Int) -> Run {
            var start = run.start
            var position = run.replyStart
            var steps = run.steps
            while position > 0 {
                if start - 1 >= free, soundsLike(start - 1, position - 1) {
                    start -= 1
                } else if start - 2 >= free, splits(start - 2, position - 1) {
                    start -= 2
                } else {
                    break
                }
                position -= 1
                steps += 1
            }
            return Run(start: start, end: run.end, matched: run.matched, steps: steps, replyStart: position, replyEnd: run.replyEnd)
        }

        // MARK: Judging

        /// Accounted for by the reply: inside a stretch that follows it,
        /// or a distinctive word it says somewhere. A common word out of
        /// step is not, since nearly every reply has its "the", "you" and
        /// "what".
        func isExplained(_ index: Int) -> Bool {
            inRun[index] || (!words[index].isCommon && (!matches[index].isEmpty || soundsLikeReply[index]))
        }

        func isForeignContent(_ index: Int) -> Bool {
            !words[index].isFunctionWord && !isExplained(index)
        }

        var explainedShare: Double {
            Double(words.indices.filter(isExplained).count) / Double(words.count)
        }

        /// The transcript follows the reply somewhere, or most of its
        /// content words are the reply's.
        var resemblesReply: Bool {
            if !runs.isEmpty { return true }
            let content = words.indices.filter { !words[$0].isFunctionWord }
            guard content.count >= 2 else { return false }
            let explained = content.filter(isExplained).count
            return Double(explained) >= EchoFilter.echoShare * Double(content.count)
        }

        /// Where the user's words begin in an utterance that overlapped the
        /// reply: after the stretches of reply that open the transcript,
        /// when there is enough of them to be sure they are the reply's
        /// (three words, or two followed by an interruption); otherwise at
        /// the start. A user who opens with two of the reply's words
        /// ("tell me more", after "tell me what you need") keeps them.
        private(set) var userStart = 0

        private func findUserStart() -> Int {
            let (end, matched) = leadingRuns
            guard end > 0, end < words.count else { return end }
            return matched >= 3 || interrupts(at: end) ? end : 0
        }

        /// Where an interruption starts with nothing before it but the
        /// reply's words and words that say nothing ("stop", "milk stop",
        /// "of the milk wait"); nil when there is none.
        var interruptionAfterReply: Int? {
            for index in words.indices {
                if interrupts(at: index) { return index }
                if isForeignContent(index) { return nil }
            }
            return nil
        }

        /// An interruption starts at `index`. After other words, one that
        /// only asks for attention or disagrees ("no", "hey") must have
        /// more after it: a lone "no" at the end of the reply's words is as
        /// likely its "now" or "on" misheard.
        private func interrupts(at index: Int) -> Bool {
            guard let interruption = interruption(at: index) else { return false }
            return index == 0 || interruption.asksToStop || !interruption.endsTranscript
        }

        /// The end of the runs that open the transcript, back to back, and
        /// how many words in them match the reply outright.
        private var leadingRuns: (end: Int, matched: Int) {
            var end = 0
            var matched = 0
            for run in runs {
                guard run.start == end else { break }
                end = run.end + 1
                matched += run.matched
            }
            return (end, matched)
        }

        /// How many words open the transcript that are the reply's, for an
        /// utterance that began just after a reply was stopped part way.
        var leadingRunLength: Int {
            leadingRuns.end
        }

        /// How many words open the transcript that are the end of the
        /// reply, for an utterance that began just after it finished: words
        /// in step with it up to its last word or, when nothing follows
        /// them, nearly up to it, the recogniser having lost the last word
        /// or two. Words in step with the reply that stop short of its end
        /// and are followed by others are not its echo, which would have
        /// gone on to its end: they are the user's.
        var endingLength: Int {
            let last = filter.reply.count - 1
            guard last >= 0 else { return 0 }
            var best = 0
            // The first word may be the tail of a word the pre-roll cut
            // into, heard as something else, when two or more after it
            // follow the reply.
            for opening in 0...min(1, words.count - 1) {
                if opening == 1, isAnswer(0) { break }
                for position in filter.reply.indices where inStep(opening, position) || splits(opening, position) {
                    let run = follow(from: opening, at: position)
                    if opening == 1, run.steps < 2, run.replyEnd != last { continue }
                    if run.replyEnd == last {
                        best = max(best, run.end + 1)
                    } else if run.replyEnd >= last - EchoFilter.endingSlack, endsMisheard(after: run) {
                        best = words.count
                    }
                }
            }
            return best
        }

        /// What follows `run`, which stopped short of the reply's last
        /// word, is the rest of the reply misheard: nothing, or a word for
        /// each word left that starts like it or is as short, none of them
        /// an answer.
        private func endsMisheard(after run: Run) -> Bool {
            let after = (run.end + 1)..<words.count
            let left = (run.replyEnd + 1)..<filter.reply.count
            guard after.count == 0 || after.count == left.count else { return false }
            return zip(after, left).allSatisfy { index, position in
                let heard = words[index].form
                let due = filter.reply[position].form
                return !isAnswer(index)
                    && (heard.first == due.first || (heard.count <= 3 && due.count <= 3))
            }
        }

        /// Words from `start` open a question or a request the reply did not
        /// say: "tell me about", "what about"; a question word followed by a
        /// verb that asks ("where is", "how do"), or "how many" and the
        /// like; a question word the reply never says, with nothing before
        /// it but the reply's words ("which universities"); or, where the
        /// user's words would begin, any question word, or a verb that asks
        /// followed by its subject ("is he", "can you"). The reply's echo
        /// keeps the reply's order, which says "how plants", not "how do
        /// plants", so the first two words must not be in step with the
        /// reply, nor all the words after them a stretch of it, nor the two
        /// a word misheard in the middle of one. A verb and a pronoun turn
        /// up in a misheard echo often enough that elsewhere they count only
        /// when followed by words that are not the reply's.
        func opensQuestion(at start: Int) -> Bool {
            guard start + 1 < words.count else { return false }
            if let request = EchoFilter.requests.first(where: { phrase in
                start + phrase.count <= words.count && phrase.indices.allSatisfy { words[start + $0].form == phrase[$0] }
            }) {
                return !followsReply(from: start, count: request.count)
            }
            guard !followsReply(from: start, count: 2) else { return false }
            // A word misheard in the middle of the reply's echo ("would
            // what like to know") is not a question, though a stretch that
            // starts with one may be ("how do plants get water").
            if let run = runOf[start], runOf[start + 1] == run, runs[run].start < start { return false }
            let first = words[start].form
            let second = words[start + 1].form
            let opensUsersWords = start == 0 || start == userStart
            let followed = start + 2 < words.count
            let restIsReply = ((start + 1)..<words.count).allSatisfy { inRun[$0] }
            if EchoFilter.questionWords.contains(first) {
                if followed, EchoFilter.askingVerbs.contains(second) {
                    // A verb the reply never says, put after a question
                    // word, asks whatever follows ("how do plants get
                    // water").
                    return matches[start + 1].isEmpty || !restIsReply
                }
                if followed, first == "how", EchoFilter.howQuestions.contains(second) {
                    return !restIsReply
                }
                // A question word the reply never says, with nothing but the
                // reply's words before it, or one opening the user's words
                // followed by words that are not all the reply's.
                if matches[start].isEmpty, (0..<start).allSatisfy({ !isForeignContent($0) }) { return true }
                return opensUsersWords && followed && !restIsReply
            }
            guard followed, EchoFilter.askingVerbs.contains(first), EchoFilter.subjects.contains(second),
                  !restIsReply else { return false }
            return opensUsersWords || ((start + 2)..<words.count).contains { !inRun[$0] }
        }

        /// The `count` words from `start` are in step with the reply, one
        /// after the other.
        private func followsReply(from start: Int, count: Int) -> Bool {
            guard start + count <= words.count else { return false }
            return matches[start].contains { position in
                (1..<count).allSatisfy { inStep(start + $0, position + $0) }
            }
        }

        /// Words that say something follow a stretch in step with the
        /// reply up to its last word: nothing the reply had said could have
        /// been heard after that ("I need to know his email", over a reply
        /// ending "what you need to know").
        var goesPastReply: Bool {
            guard let run = runs.last(where: { $0.replyEnd == filter.reply.count - 1 }) else { return false }
            let after = (run.end + 1)..<words.count
            return after.count >= 2 && after.contains(where: isForeignContent)
        }

        /// A word that answers or interrupts rather than repeats: "yes",
        /// "no", "why", "stop".
        private func isAnswer(_ index: Int) -> Bool {
            let form = words[index].form
            return EchoFilter.answers.contains(form) || EchoFilter.questionWords.contains(form)
                || EchoFilter.interruptionLength(in: words, at: index) != nil
        }

        struct Interruption {
            /// Ends with the end of the transcript.
            let endsTranscript: Bool
            let asksToStop: Bool
        }

        /// Words from `start` open with "stop", "wait", "hold on", "no" and
        /// the like, perhaps after an "oh" or an "okay", that are not the
        /// reply's: not in step with it, and either not all words it says
        /// somewhere, or that sound the same ("wait" for its "weight", "no"
        /// for its "know"), or said twice over ("stop stop"). One that asks
        /// for attention or disagrees ("hey", "no", "sorry") counts only on
        /// its own or followed by words that do not carry on the reply: the
        /// recogniser makes up an "okay" or a "hey" at the start of an
        /// echo now and then.
        func interruption(at start: Int) -> Interruption? {
            var end = start
            while end < words.count, EchoFilter.fillers.contains(words[end].form) {
                end += 1
            }
            let phraseStart = end
            var phrases: [ArraySlice<Word>] = []
            while end < words.count, let length = EchoFilter.interruptionLength(in: words, at: end) {
                phrases.append(words[end..<(end + length)])
                end += length
            }
            guard end > phraseStart else { return nil }
            guard (start..<end).allSatisfy({ !inRun[$0] }) else { return nil }
            let repeated = phrases.count >= 2 && phrases.dropFirst().allSatisfy {
                $0.map(\.form) == phrases[0].map(\.form)
            }
            let unsaid = (phraseStart..<end).contains { matches[$0].isEmpty && !filter.saysHomophone(of: words[$0]) }
            guard repeated || unsaid else { return nil }
            let asksToStop = phrases.contains { EchoFilter.stopRequests.contains($0.map(\.form)) }
            if !asksToStop, end < words.count, inRun[end] { return nil }
            return Interruption(endsTranscript: end == words.count, asksToStop: asksToStop)
        }

        /// The longest stretch of consecutive words in `range` the reply
        /// does not account for that says something: it holds two content
        /// words, or opens with a question word ("what was that"). 0 when
        /// there is none.
        func longestForeignStretch(in range: Range<Int>) -> Int {
            var longest = 0
            var length = 0
            var content = 0
            var opensWithQuestion = false
            for index in range {
                guard !isExplained(index) else {
                    length = 0
                    content = 0
                    continue
                }
                if length == 0 {
                    opensWithQuestion = EchoFilter.questionWords.contains(words[index].form)
                }
                length += 1
                if !words[index].isFunctionWord { content += 1 }
                if content >= 2 || opensWithQuestion { longest = max(longest, length) }
            }
            return longest
        }

        /// The transcript from word `index` on, as it was written.
        func text(from index: Int) -> String {
            let text = index == 0 ? Substring(transcript) : transcript[words[index].start...]
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    // MARK: - Matching words

    /// The reply positions `word` matches outright: the same word, or a
    /// distinctive word of the reply's one letter in four away from it.
    private func positions(matching word: Word) -> Set<Int> {
        var found = Set(positions[word.key] ?? [])
        guard !word.isCommon else { return found }
        for other in distinctive where other.key != word.key && Self.isMisheard(other.keyLetters, word.keyLetters) {
            found.formUnion(positions[other.key] ?? [])
        }
        return found
    }

    /// The reply says a word that sounds the same as `word`.
    private func saysHomophone(of word: Word) -> Bool {
        word.homophoneGroup.map(homophonesSaid.contains) ?? false
    }

    /// `word` sounds so like a distinctive word of the reply's that it can
    /// be that word wherever it is, or it and a neighbour are one of them
    /// heard as two.
    private func isLooselyInReply(_ word: Word, next: Word?, previous: Word?) -> Bool {
        guard !word.isCommon else { return false }
        let withNext = next.map { word.letters + $0.letters }
        let withPrevious = previous.map { $0.letters + word.letters }
        for other in distinctive {
            if word.sounds.count >= 3, word.sounds == other.sounds { return true }
            for joined in [withNext, withPrevious] {
                if let joined, Self.isSimilar(joined, other.letters, atLeast: Self.misheardSimilarity) { return true }
            }
        }
        return word.letters.count >= 4
            && longCommon.contains { Self.isSimilar(word.letters, $0, atLeast: Self.misheardSimilarity) }
    }

    /// Two different distinctive words close enough to be one misheard. Short
    /// words have to match exactly: one letter changes too many of them
    /// into other words.
    private static func isMisheard(_ first: [Character], _ second: [Character]) -> Bool {
        min(first.count, second.count) >= 4 && isSimilar(first, second, atLeast: misheardSimilarity)
    }

    /// `heard` can be the reply's word `due` heard wrong, given that the
    /// reply's word due there is `due`: the same word, a word that sounds
    /// the same ("no" for "know"), a short word one letter off ("of" for
    /// "or"), or a word spelt or sounding much like it ("fillings" for
    /// "feelings", "Jong" for "Zhang").
    private static func soundsAlike(_ heard: Word, _ due: Word) -> Bool {
        if heard.key == due.key { return true }
        if let group = heard.homophoneGroup, group == due.homophoneGroup { return true }
        let a = heard.letters
        let b = due.letters
        if heard.isCommon || due.isCommon {
            if heard.isCommon && due.isCommon {
                return a.count == b.count && a.count >= 2 && editDistance(a, b) == 1
            }
            return min(a.count, b.count) >= 4 && isSimilar(a, b, atLeast: misheardSimilarity)
        }
        guard min(a.count, b.count) >= 3 else { return false }
        return isSimilar(a, b, atLeast: nearSimilarity) || (heard.sounds.count >= 2 && heard.sounds == due.sounds)
    }

    /// The spellings are at least `threshold` alike: 1 for the same, less
    /// a share for each letter changed, added or dropped, relative to the
    /// longer word. The difference in length alone rules most pairs out.
    private static func isSimilar(_ a: [Character], _ b: [Character], atLeast threshold: Double) -> Bool {
        let longer = Double(max(a.count, b.count))
        guard !a.isEmpty, !b.isEmpty, 1 - Double(abs(a.count - b.count)) / longer >= threshold else { return false }
        return 1 - Double(editDistance(a, b)) / longer >= threshold
    }

    private static func editDistance(_ a: [Character], _ b: [Character]) -> Int {
        var previous = Array(0...b.count)
        var current = Array(repeating: 0, count: b.count + 1)
        for i in 1...a.count {
            current[0] = i
            for j in 1...b.count {
                let substitution = previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)
                current[j] = min(substitution, previous[j] + 1, current[j - 1] + 1)
            }
            swap(&previous, &current)
        }
        return previous[b.count]
    }

    /// A word's consonant sounds, roughly: letters that sound alike share a
    /// digit, as in Soundex, vowels and "h", "w", "y" are dropped, and a
    /// sound written twice in a row counts once. "Lin" and "Lynn" share
    /// one, and so do "Zhang" and "Jong".
    private static func skeleton(_ letters: [Character]) -> [Character] {
        var sounds: [Character] = []
        var previous: Character?
        for letter in letters {
            let sound: Character?
            switch letter {
            case "b", "f", "p", "v": sound = "1"
            case "c", "g", "j", "k", "q", "s", "x", "z": sound = "2"
            case "d", "t": sound = "3"
            case "l": sound = "4"
            case "m", "n": sound = "5"
            case "r": sound = "6"
            default: sound = nil
            }
            if let sound, sound != previous { sounds.append(sound) }
            previous = sound
        }
        return sounds
    }

    // MARK: - Words

    private struct Word {
        /// Lowercased, apostrophes dropped ("don't" and "dont" are one
        /// word), numbers and symbols spelled out as the synthesizer says
        /// them.
        let form: String
        /// What words are compared by: `form`, with a distinctive word's
        /// plural and verb endings taken off.
        let key: String
        /// Carries no meaning of its own ("the", "you", "have").
        let isFunctionWord: Bool
        /// A function word or a question word: one so frequent that the
        /// reply saying it somewhere says nothing about where this one
        /// came from. Only the rest are matched loosely.
        let isCommon: Bool
        /// Where the word starts in the text it came from. The words a
        /// number or a symbol is spelled out into all start at it.
        let start: String.Index
        /// `form` and `key` letter by letter, and `form`'s consonant
        /// sounds, worked out once since words are compared many times.
        let letters: [Character]
        let keyLetters: [Character]
        let sounds: [Character]
        /// The group in `homophones` the word belongs to.
        let homophoneGroup: Int?

        init(_ form: String, start: String.Index) {
            self.form = form
            isFunctionWord = EchoFilter.functionWords.contains(form)
            isCommon = isFunctionWord || EchoFilter.questionWords.contains(form)
            key = isCommon ? form : EchoFilter.stem(form)
            self.start = start
            letters = Array(form)
            keyLetters = Array(key)
            sounds = EchoFilter.skeleton(letters)
            homophoneGroup = EchoFilter.homophones[form]
        }
    }

    private static func words(in text: String) -> [Word] {
        var words: [Word] = []
        var token = ""
        var start: String.Index?
        func endToken() {
            guard let begun = start else { return }
            for form in spoken(token) {
                words.append(Word(form, start: begun))
            }
            token = ""
            start = nil
        }
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if character.isLetter || character.isNumber {
                if start == nil { start = index }
                token += character.lowercased()
            } else if start != nil, apostrophes.contains(character) {
                // Inside a word: skipped, and the word goes on.
            } else {
                endToken()
                for form in symbols[character] ?? [] {
                    words.append(Word(form, start: index))
                }
            }
            index = text.index(after: index)
        }
        endToken()
        // "°F" is said "degrees Fahrenheit".
        for index in words.indices.dropFirst() where words[index - 1].form == "degrees" {
            if let scale = scales[words[index].form] {
                words[index] = Word(scale, start: words[index].start)
            }
        }
        return words
    }

    /// A token as the synthesizer says it: a number as its words, "ok" as
    /// "okay", anything else as it is.
    private static func spoken(_ token: String) -> [String] {
        if token == "ok" { return ["okay"] }
        guard token.allSatisfy(\.isASCII), token.allSatisfy(\.isNumber), let number = Int(token),
              token.count <= 4, token.count == 1 || !token.hasPrefix("0") else { return [token] }
        return spoken(number: number)
    }

    /// 0 to 9999 in words, years in pairs ("eighteen eighty nine").
    private static func spoken(number: Int) -> [String] {
        switch number {
        case 0..<20:
            return [ones[number]]
        case 20..<100:
            return [tens[number / 10]] + (number % 10 == 0 ? [] : [ones[number % 10]])
        case 100..<1000:
            return [ones[number / 100], "hundred"] + (number % 100 == 0 ? [] : spoken(number: number % 100))
        case 1100..<2000 where number % 100 != 0, 2010..<2100:
            return spoken(number: number / 100) + (number % 100 < 10 ? ["oh", ones[number % 100]] : spoken(number: number % 100))
        default:
            return spoken(number: number / 1000) + ["thousand"] + (number % 1000 == 0 ? [] : spoken(number: number % 1000))
        }
    }

    private static let ones = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    private static let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    /// Symbols the synthesizer reads out as words.
    private static let symbols: [Character: [String]] = [
        "%": ["percent"], "&": ["and"], "+": ["plus"], "=": ["equals"], "°": ["degrees"],
    ]
    private static let scales: [String: String] = ["f": "fahrenheit", "c": "celsius"]

    /// Plural and verb endings, taken off so "lists" meets "list" and
    /// "feelings" meets "feeling". Crude, but applied to both sides alike.
    private static func stem(_ form: String) -> String {
        var word = form
        if word.count > 4, word.hasSuffix("ies") {
            word = String(word.dropLast(3)) + "y"
        } else if word.count > 3, word.hasSuffix("s"),
                  !word.hasSuffix("ss"), !word.hasSuffix("us"), !word.hasSuffix("is") {
            word.removeLast()
        }
        if word.count > 5, word.hasSuffix("ing") {
            word.removeLast(3)
        } else if word.count > 4, word.hasSuffix("ed"), !word.hasSuffix("eed") {
            word.removeLast(2)
        }
        return word
    }

    /// The length of the interruption ("stop", "hold on") at `index`, if
    /// one starts there.
    private static func interruptionLength(in words: [Word], at index: Int) -> Int? {
        for phrase in interruptions where index + phrase.count <= words.count {
            if zip(phrase, words[index...]).allSatisfy({ $0 == $1.form }) {
                return phrase.count
            }
        }
        return nil
    }

    /// Asking the reply to stop or wait.
    private static let stopRequests: Set<[String]> = [
        ["hold", "on"], ["hold", "up"], ["hang", "on"], ["shut", "up"], ["be", "quiet"], ["never", "mind"],
        ["excuse", "me"], ["stop"], ["wait"], ["pause"], ["cancel"], ["quiet"], ["enough"], ["nevermind"],
    ]

    /// What people say to cut a speaker off: the stop requests, and words
    /// that ask for something else or disagree. Longer phrases come first
    /// so "hold on" is not read as an unknown "hold".
    private static let interruptions: [[String]] = stopRequests.sorted { $0.count > $1.count } + [
        ["say", "that", "again"], ["repeat", "that"], ["say", "again"],
        ["no"], ["nope"], ["hey"], ["actually"], ["sorry"],
        ["louder"], ["slower"], ["faster"], ["next"], ["skip"],
    ]

    /// Said before an interruption without changing it: "oh wait",
    /// "okay stop", "please stop".
    private static let fillers: Set<String> = ["um", "uh", "er", "oh", "ah", "hmm", "well", "so", "please", "okay"]

    /// Short answers to a reply's closing question, function words though
    /// they are.
    private static let answers: Set<String> = [
        "yes", "yeah", "yep", "yup", "no", "nope", "sure", "okay", "please", "thanks", "right", "correct",
    ]

    /// Words a recogniser writes one way for another that sounds the same,
    /// each mapped to its group.
    private static let homophones: [String: Int] = {
        let groups: [[String]] = [
            ["no", "know", "now"], ["wait", "weight"], ["right", "write"], ["there", "their", "theyre"],
            ["to", "too", "two"], ["for", "four"], ["by", "buy", "bye"], ["here", "hear"], ["one", "won"],
            ["see", "sea"], ["be", "bee"], ["new", "knew"], ["our", "hour"], ["hey", "hay"],
            ["whole", "hole"], ["would", "wood"], ["eight", "ate"], ["i", "eye", "ai"],
        ]
        var groupOf: [String: Int] = [:]
        for (group, words) in groups.enumerated() {
            for word in words { groupOf[word] = group }
        }
        return groupOf
    }()

    private static let apostrophes: Set<Character> = ["'", "\u{2019}", "\u{2018}", "\u{02BC}", "`", "\u{00B4}"]

    /// Common, but not function words: "what about Berlin" is a question
    /// because of its "what".
    private static let questionWords: Set<String> = [
        "what", "whats", "who", "whos", "whom", "whose", "which",
        "how", "hows", "why", "where", "wheres", "when",
    ]

    /// Verbs that open a question about something: "is it", "can you".
    private static let askingVerbs: Set<String> = [
        "is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should",
        "has", "have", "had", "isnt", "arent", "doesnt", "dont", "didnt", "cant", "wont",
    ]

    /// What follows an asking verb in a question: "is he", "can you".
    private static let subjects: Set<String> = [
        "i", "you", "he", "she", "it", "we", "they", "there", "that", "this", "these", "those",
    ]

    /// Asking for something without a question word.
    private static let requests: [[String]] = [
        ["tell", "me", "about"], ["tell", "me", "more"], ["what", "about"], ["how", "about"],
    ]

    /// "how many", "how long".
    private static let howQuestions: Set<String> = [
        "many", "much", "long", "far", "old", "big", "tall", "often",
    ]

    /// Words in nearly every sentence, which on their own say nothing
    /// about who is talking.
    private static let functionWords: Set<String> = [
        "a", "an", "the",
        "i", "im", "ive", "id", "ill", "me", "my", "mine", "myself",
        "you", "youre", "youve", "youd", "youll", "your", "yours", "yourself",
        "we", "were", "weve", "wed", "well", "us", "our", "ours",
        "he", "hes", "him", "his", "she", "shes", "her", "hers",
        "it", "its", "itself", "they", "theyre", "them", "their", "theirs",
        "this", "that", "thats", "these", "those", "there", "theres", "here", "heres",
        "is", "isnt", "am", "are", "arent", "was", "wasnt", "werent", "be", "been", "being",
        "do", "does", "doesnt", "did", "didnt", "dont",
        "have", "has", "hasnt", "had", "hadnt", "havent",
        "can", "cant", "cannot", "could", "couldnt", "will", "wont", "would", "wouldnt",
        "should", "shouldnt", "may", "might", "must",
        "to", "of", "in", "on", "at", "by", "for", "with", "from", "into", "about", "as",
        "than", "then", "and", "or", "but", "if", "so", "not", "no", "nor",
        "just", "also", "very", "too", "some", "any", "all", "each", "every", "both", "such",
        "up", "out", "um", "uh", "oh", "ah", "hmm", "yeah", "yes", "okay", "please", "let", "lets",
    ]
}
