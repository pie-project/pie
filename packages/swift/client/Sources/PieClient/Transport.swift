import Foundation

/// Carries a `PieClient`'s MessagePack frames both ways.
public protocol PieTransport: AnyObject, Sendable {
    /// Every frame the server sends, in order; finishes when the transport closes.
    var frames: AsyncThrowingStream<Data, Error> { get }
    func send(_ frame: Data) async throws
    func close()
}

/// A `pie serve` gateway at `ws://host:port` (its `/v1/ws` route when the
/// URL names no path). `identity` is the `x-pie-identity` header
/// (`tenant/user`) the gateway requires of a client no edge proxy fronts.
public final class WebSocketTransport: PieTransport, @unchecked Sendable {
    private let task: URLSessionWebSocketTask
    public let frames: AsyncThrowingStream<Data, Error>
    private let continuation: AsyncThrowingStream<Data, Error>.Continuation

    public init(url: URL, identity: String? = "default/swift", session: URLSession = .shared) {
        var url = url
        if url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/")).isEmpty {
            url = url.appending(path: "v1/ws")
        }
        var request = URLRequest(url: url)
        if let identity { request.setValue(identity, forHTTPHeaderField: "x-pie-identity") }
        task = session.webSocketTask(with: request)
        task.maximumMessageSize = 64 << 20
        (frames, continuation) = AsyncThrowingStream.makeStream()
        task.resume()
        receive()
    }

    deinit {
        task.cancel(with: .normalClosure, reason: nil)
    }

    public func send(_ frame: Data) async throws {
        try await task.send(.data(frame))
    }

    public func close() {
        task.cancel(with: .normalClosure, reason: nil)
        continuation.finish()
    }

    private func receive() {
        task.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                self.continuation.finish(throwing: error)
            case .success(.data(let data)):
                self.continuation.yield(data)
                self.receive()
            case .success:
                self.receive()
            }
        }
    }
}
