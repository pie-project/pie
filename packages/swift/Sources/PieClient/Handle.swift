import Foundation

/// State this session owns between processes. Lazy: nothing exists on the
/// server until a process binds it, so making one costs a name.
public protocol Handle: Sendable {
    var client: PieClient { get }
    var name: UInt64 { get }
    init(client: PieClient, name: UInt64)
    /// What the input carries where this handle goes, under the name `slot`
    /// it moves in by.
    func marker(_ slot: String) -> [String: String]
}

extension Handle {
    public func marker(_ slot: String) -> [String: String] {
        ["$handle": slot]
    }

    /// An independent copy of this handle as it stands.
    public func fork() async throws -> Self {
        let child = await client.nextHandle()
        _ = try await client.handleOp(.fork(name: child, from: name))
        return Self(client: client, name: child)
    }

    /// The handle's state as JSON; its fields depend on what it holds.
    public func stat() async throws -> Data {
        try await client.handleOp(.stat(name: name))
    }

    /// Merges `props` for the worker policy to read while this rests; a nil
    /// removes one (`[String: Int?]`; `setProps(json:)` for mixed values).
    /// Sent while a process has it, they land with it.
    public func setProps(_ props: some Encodable) async throws {
        try await setProps(json: JSONEncoder().encode(props))
    }

    /// `setProps` with `json` an object's text; a null removes one.
    public func setProps(json: Data) async throws {
        _ = try await client.handleOp(.setProps(name: name, props: String(decoding: json, as: UTF8.self)))
    }

    public func close() async throws {
        _ = try await client.handleOp(.close(name: name))
    }
}

/// One sequence's model state.
public struct WorkingSet: Handle {
    public let client: PieClient
    public let name: UInt64

    public init(client: PieClient, name: UInt64) {
        self.client = client
        self.name = name
    }

    /// Keeps this working set at rest under `snapshot`, past this session and
    /// this server, and says what it kept. It stays as it was.
    @discardableResult
    public func save(_ snapshot: String) async throws -> Snapshot {
        try await client.save(name, snapshot: snapshot)
    }
}

/// A grammar match in progress. It stands where it stood across processes
/// with the working set it constrains: move and fork them together.
public struct Matcher: Handle {
    public let client: PieClient
    public let name: UInt64

    public init(client: PieClient, name: UInt64) {
        self.client = client
        self.name = name
    }

    public func marker(_ slot: String) -> [String: String] {
        ["$handle": slot, "kind": "matcher"]
    }
}

/// Host memory a process made or a client uploaded: it moves on to another
/// process without crossing the client, and comes down only when asked.
public protocol Media: Handle {}

extension Media {
    /// The whole of this handle, encoded as `format`. It stays.
    public func read(format: String) async throws -> Data {
        try await client.read(name, format: format)
    }
}

/// Pictures, or one: `count` frames of RGB8.
public struct Frames: Media {
    public let client: PieClient
    public let name: UInt64

    public init(client: PieClient, name: UInt64) {
        self.client = client
        self.name = name
    }

    public func marker(_ slot: String) -> [String: String] {
        ["$handle": slot, "kind": "frames"]
    }
}

/// Sound: interleaved samples at a rate.
public struct Pcm: Media {
    public let client: PieClient
    public let name: UInt64

    public init(client: PieClient, name: UInt64) {
        self.client = client
        self.name = name
    }

    public func marker(_ slot: String) -> [String: String] {
        ["$handle": slot, "kind": "pcm"]
    }
}

/// A working set kept by `WorkingSet.save`.
public struct Snapshot: Sendable, Equatable, Decodable {
    public let name: String
    public let tokens: Int
    public let bytes: Int
}
