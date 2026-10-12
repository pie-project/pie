import Foundation

public enum PieError: Error, Sendable, Equatable, LocalizedError {
    /// The connection or session is closed.
    case connectionClosed
    /// The server refused a request (an unknown inferlet, a bad input, ...).
    case requestFailed(String)
    /// The inferlet ended with an error.
    case processFailed(String)
    /// The in-process server failed (boot, install, a closed server, ...).
    case server(String)
    /// The server sent a frame this client cannot read.
    case malformedFrame(String)

    public var errorDescription: String? {
        switch self {
        case .connectionClosed: "The connection is closed."
        case .requestFailed(let reason): "The server refused the request: \(reason)"
        case .processFailed(let reason): "The inferlet failed: \(reason)"
        case .server(let reason): reason
        case .malformedFrame(let reason): "Malformed frame: \(reason)"
        }
    }
}
