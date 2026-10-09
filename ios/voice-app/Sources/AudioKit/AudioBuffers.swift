import AVFoundation
import Foundation

/// Loudness arithmetic shared by the meters and the voice-activity
/// detector, so "0.6 on the orb" means the same thing for the microphone
/// and for the reply.
enum AudioLevel {

    /// dBFS of a root-mean-square amplitude. Digital silence maps to
    /// -140 dB rather than minus infinity.
    static func decibels(rms: Float) -> Float {
        20 * log10(max(rms, 1e-7))
    }

    /// Maps dBFS onto 0...1 for the meters: -50 dBFS is a quiet room,
    /// 0 dBFS is clipping.
    static func normalized(decibels: Float) -> Float {
        max(0, min(1, (decibels + 50) / 50))
    }

    /// RMS of each consecutive run of `windowFrames` frames of the first
    /// channel; the last window may be shorter. Float32 and Int16 buffers,
    /// interleaved or not.
    static func windowRMS(_ buffer: AVAudioPCMBuffer, windowFrames: Int) -> [Float] {
        let frames = Int(buffer.frameLength)
        guard frames > 0, windowFrames > 0 else { return [] }
        let stride = buffer.stride
        var result: [Float] = []
        result.reserveCapacity(frames / windowFrames + 1)

        if let samples = buffer.floatChannelData?[0] {
            var start = 0
            while start < frames {
                let end = min(start + windowFrames, frames)
                var sum: Float = 0
                for frame in start..<end {
                    let sample = samples[frame * stride]
                    sum += sample * sample
                }
                result.append(sqrt(sum / Float(end - start)))
                start = end
            }
        } else if let samples = buffer.int16ChannelData?[0] {
            let scale = 1 / Float(Int16.max)
            var start = 0
            while start < frames {
                let end = min(start + windowFrames, frames)
                var sum: Float = 0
                for frame in start..<end {
                    let sample = Float(samples[frame * stride]) * scale
                    sum += sample * sample
                }
                result.append(sqrt(sum / Float(end - start)))
                start = end
            }
        }
        return result
    }
}

extension AVAudioPCMBuffer {

    /// A deep copy. Tap buffers belong to the engine and are reused once
    /// the tap block returns, so anything kept past that (the pre-roll,
    /// audio held for a recogniser restart) has to be copied.
    func copied() -> AVAudioPCMBuffer? {
        guard let copy = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(frameLength, 1)) else {
            return nil
        }
        copy.frameLength = frameLength
        let source = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: audioBufferList))
        let destination = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        for (from, to) in zip(source, destination) {
            guard let fromData = from.mData, let toData = to.mData else { continue }
            memcpy(toData, fromData, Int(min(from.mDataByteSize, to.mDataByteSize)))
        }
        return copy
    }

    /// The buffer cut into consecutive pieces of at most `frames` frames.
    func split(maxFrames frames: AVAudioFrameCount) -> [AVAudioPCMBuffer] {
        guard frameLength > frames, frames > 0 else { return [self] }
        let bytesPerFrame = Int(format.streamDescription.pointee.mBytesPerFrame)
        let source = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: audioBufferList))
        var pieces: [AVAudioPCMBuffer] = []
        var offset: AVAudioFrameCount = 0
        while offset < frameLength {
            let count = min(frames, frameLength - offset)
            guard let piece = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count) else { break }
            piece.frameLength = count
            let destination = UnsafeMutableAudioBufferListPointer(piece.mutableAudioBufferList)
            for (from, to) in zip(source, destination) {
                guard let fromData = from.mData, let toData = to.mData else { continue }
                memcpy(toData, fromData + Int(offset) * bytesPerFrame, Int(count) * bytesPerFrame)
            }
            pieces.append(piece)
            offset += count
        }
        return pieces
    }

    var duration: TimeInterval {
        Double(frameLength) / format.sampleRate
    }
}

/// Converts a stream of buffers into one fixed format.
///
/// One instance per stream: the resampler keeps filter state between
/// calls, which is what lets a stream converted piece by piece come out
/// without a click at every seam. `finish()` drains the samples the
/// filter is still holding at the end of the stream.
///
/// The resampler runs at its best, the mastering algorithm at maximum
/// quality. The synthesizer's voices render at 22.05 kHz and the playback
/// format is 48 kHz. Measured with white noise at 22.05 kHz, the default
/// filter starts rolling off at 9 kHz: -4.5 dB at 9.5 to 10 kHz, -8 to
/// -23 dB from 10 kHz to the voice's 11 kHz limit, which is the "s" and
/// "t" and the air of a voice, dulled. Mastering at maximum quality is
/// flat to 10.75 kHz and blocks the mirror images just above 11 kHz that
/// the default lets through at -34 dB. Converted piece by piece the way
/// the synthesizer delivers it, its output has the same length as the
/// default's, sample for sample the same as converting the sentence in
/// one go, with no added delay. It costs ten times the arithmetic, which
/// at a voice's data rate is still about 0.6% of real time on a Mac.
final class PCMConverter {
    let outputFormat: AVAudioFormat
    private var converter: AVAudioConverter?

    init(outputFormat: AVAudioFormat) {
        self.outputFormat = outputFormat
    }

    func convert(_ input: AVAudioPCMBuffer) -> [AVAudioPCMBuffer] {
        guard input.frameLength > 0 else { return [] }
        if converter?.inputFormat != input.format {
            converter = AVAudioConverter(from: input.format, to: outputFormat)
            // The synthesizer's voices are mono today; a stereo voice must
            // still fit the mono playback channel.
            converter?.downmix = true
            if input.format.sampleRate != outputFormat.sampleRate {
                converter?.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
                converter?.sampleRateConverterQuality = AVAudioQuality.max.rawValue
            }
        }
        guard let converter else { return [] }
        var supplied = false
        // Room for the whole conversion at once, so a buffer comes out as
        // one buffer rather than a run of fixed-size fragments.
        let ratio = outputFormat.sampleRate / input.format.sampleRate
        let expected = AVAudioFrameCount((Double(input.frameLength) * ratio).rounded(.up)) + 1024
        return drain(converter, capacity: expected) { _, status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return input
        }
    }

    func finish() -> [AVAudioPCMBuffer] {
        guard let converter else { return [] }
        self.converter = nil
        return drain(converter, capacity: 4096) { _, status in
            status.pointee = .endOfStream
            return nil
        }
    }

    private func drain(
        _ converter: AVAudioConverter,
        capacity: AVAudioFrameCount,
        input: @escaping AVAudioConverterInputBlock
    ) -> [AVAudioPCMBuffer] {
        var output: [AVAudioPCMBuffer] = []
        while let buffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) {
            var error: NSError?
            let status = converter.convert(to: buffer, error: &error, withInputFrom: input)
            if buffer.frameLength > 0 {
                output.append(buffer)
            }
            // `.haveData` means the output filled up with more to come;
            // anything else (ran dry, end of stream, error) is the end of
            // what this call can produce.
            guard status == .haveData, buffer.frameLength > 0 else { break }
        }
        return output
    }
}

/// Gathers a stream of small buffers into fixed-size ones.
///
/// The synthesizer renders in pieces of about 11 ms. Scheduled as they
/// come, that is close to a hundred player buffers, completion callbacks
/// and main-queue hops for every second of speech; gathered into 100 ms
/// buffers it is ten. Deinterleaved float formats only, which is what the
/// playback format is.
final class BufferCoalescer {
    let format: AVAudioFormat
    let chunkFrames: AVAudioFrameCount
    private var pending: AVAudioPCMBuffer?

    init(format: AVAudioFormat, chunkFrames: AVAudioFrameCount) {
        self.format = format
        self.chunkFrames = chunkFrames
    }

    /// Takes `buffers` and returns every chunk they complete.
    func append(_ buffers: [AVAudioPCMBuffer]) -> [AVAudioPCMBuffer] {
        var complete: [AVAudioPCMBuffer] = []
        for buffer in buffers {
            var offset: AVAudioFrameCount = 0
            while offset < buffer.frameLength {
                guard let chunk = pending ?? AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunkFrames) else {
                    return complete
                }
                let count = min(chunkFrames - chunk.frameLength, buffer.frameLength - offset)
                copy(count, from: buffer, at: offset, into: chunk)
                offset += count
                if chunk.frameLength == chunkFrames {
                    complete.append(chunk)
                    pending = nil
                } else {
                    pending = chunk
                }
            }
        }
        return complete
    }

    /// Whatever is left, as a final shorter chunk.
    func flush() -> [AVAudioPCMBuffer] {
        defer { pending = nil }
        guard let pending, pending.frameLength > 0 else { return [] }
        return [pending]
    }

    private func copy(
        _ count: AVAudioFrameCount,
        from source: AVAudioPCMBuffer,
        at offset: AVAudioFrameCount,
        into chunk: AVAudioPCMBuffer
    ) {
        guard let from = source.floatChannelData, let to = chunk.floatChannelData else { return }
        let channels = Int(min(source.format.channelCount, chunk.format.channelCount))
        for channel in 0..<channels {
            (to[channel] + Int(chunk.frameLength)).update(from: from[channel] + Int(offset), count: Int(count))
        }
        chunk.frameLength += count
    }
}

extension AVAudioFile {

    /// The whole file, converted to `format` and cut into pieces of about
    /// `pieceDuration`. Short pieces mean a piece cut off by an engine
    /// rebuild and replayed costs only a fraction of a second.
    static func loadPieces(
        of url: URL,
        as format: AVAudioFormat,
        pieceDuration: TimeInterval
    ) throws -> [AVAudioPCMBuffer] {
        let file = try AVAudioFile(forReading: url)
        let length = AVAudioFrameCount(file.length)
        guard length > 0,
              let whole = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: length)
        else { return [] }
        try file.read(into: whole)

        let converter = PCMConverter(outputFormat: format)
        let converted = converter.convert(whole) + converter.finish()
        let pieceFrames = AVAudioFrameCount(pieceDuration * format.sampleRate)
        return converted.flatMap { $0.split(maxFrames: pieceFrames) }
    }
}
