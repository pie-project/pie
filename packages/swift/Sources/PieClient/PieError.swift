import Foundation

public enum PieError: Error, Sendable, Equatable, LocalizedError {
    /// The connection or session is closed.
    case connectionClosed
    /// The server refused a request (an unknown inferlet, a bad input, ...).
    case requestFailed(String)
    /// A handle op, a launch or a read was refused, with its code when the
    /// server named one.
    case handleFailed(HandleErrorCode?, String)
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
        case .handleFailed(_, let reason): "The server refused the handle: \(reason)"
        case .processFailed(let reason): "The inferlet failed: \(reason)"
        case .server(let reason): reason
        case .malformedFrame(let reason): "Malformed frame: \(reason)"
        }
    }
}

/// Why the server refused a handle, for a client to branch on.
public enum HandleErrorCode: String, Sendable, Equatable {
    /// No such name in this session.
    case unknown
    /// It existed and is gone; the reason says why.
    case dead
    /// A session or worker limit: closing a handle makes room.
    case quota
    /// The name is taken, or the handle is in a process.
    case inUse = "in_use"
    /// The request could never work, whatever the session held.
    case invalid

    /// The code a refusal's text starts with, if it is one.
    init?(refusal reason: String) {
        self.init(rawValue: String(reason.prefix { $0 != ":" }))
    }
}
