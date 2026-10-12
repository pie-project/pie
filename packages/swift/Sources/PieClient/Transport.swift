import Foundation

/// Carries a `PieClient`'s MessagePack frames both ways.
public protocol PieTransport: Sendable {
    /// Every frame the server sends, in order; finishes when the transport closes.
    var frames: AsyncThrowingStream<Data, any Error> { get }
    func send(_ frame: Data) async throws
    func close()
}

/// A `pie serve` gateway at `ws://host:port` (its `/v1/ws` route when the
/// URL names no path). `identity` is the `x-pie-identity` header
/// (`tenant/user`) the gateway requires of a client no edge proxy fronts.
public final class WebSocketTransport: PieTransport {
    public let frames: AsyncThrowingStream<Data, any Error>
    private let task: URLSessionWebSocketTask
    private let continuation: AsyncThrowingStream<Data, any Error>.Continuation

    public init(url: URL, identity: String? = "default/swift", session: URLSession = .shared) {
        var url = url
        if url.path().trimmingCharacters(in: CharacterSet(charactersIn: "/")).isEmpty {
            url.append(path: "v1/ws")
        }
        var request = URLRequest(url: url)
        request.setValue(identity, forHTTPHeaderField: "x-pie-identity")
        let task = session.webSocketTask(with: request)
        task.maximumMessageSize = 64 << 20
        let (frames, continuation) = AsyncThrowingStream.makeStream(of: Data.self)
        self.task = task
        self.frames = frames
        self.continuation = continuation
        task.resume()
        Task {
            do {
                while true {
                    if case .data(let frame) = try await task.receive() {
                        continuation.yield(frame)
                    }
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    deinit {
        close()
    }

    public func send(_ frame: Data) async throws {
        try await task.send(.data(frame))
    }

    public func close() {
        continuation.finish()
        task.cancel(with: .normalClosure, reason: nil)
    }
}
