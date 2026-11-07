import Foundation

/// crates/client-api's frames, as far as this client speaks them.
enum ClientFrame {
    case ping
    case launch(inferlet: String, input: String, captureOutputs: Bool, handles: [String: UInt64], returns: [String: UInt64])
    case signal(processID: String, message: String)
    case terminate(processID: String)
    case handles(HandleOp)
    case readHandle(name: UInt64, format: String, reader: String)

    func encoded(correlationID: UInt32) -> Data {
        let id = MessagePack.int(Int64(correlationID))
        let fields: [String: MessagePack] = switch self {
        case .ping:
            ["type": .string("ping"), "corr_id": id]
        case .launch(let inferlet, let input, let captureOutputs, let handles, let returns):
            ["type": .string("launch_process"), "corr_id": id, "inferlet": .string(inferlet),
             "input": .string(input), "capture_outputs": .bool(captureOutputs),
             "handles": Self.names(handles), "returns": Self.names(returns)]
        case .signal(let processID, let message):
            ["type": .string("signal_process"), "process_id": .string(processID), "message": .string(message)]
        case .terminate(let processID):
            ["type": .string("terminate_process"), "corr_id": id, "process_id": .string(processID)]
        case .handles(let op):
            ["type": .string("handles"), "corr_id": id, "ops": .array([op.encoded])]
        case .readHandle(let name, let format, let reader):
            ["type": .string("read_handle"), "corr_id": id, "name": .int(Int64(name)),
             "format": .string(format), "reader": .string(reader)]
        }
        return MessagePack.map(fields).encoded()
    }

    private static func names(_ handles: [String: UInt64]) -> MessagePack {
        .map(handles.mapValues { MessagePack.int(Int64($0)) })
    }

    /// Whether the server answers this frame with a `response`.
    var expectsResponse: Bool {
        if case .signal = self { return false }
        return true
    }
}

/// crates/client-api's `HandleOp`.
enum HandleOp: Sendable {
    case new(name: UInt64, kind: Kind)
    case fork(name: UInt64, from: UInt64)
    case close(name: UInt64)
    case stat(name: UInt64)
    /// `props` is a JSON object's text, which `Props` reads as well as a map.
    case setProps(name: UInt64, props: String)
    case save(name: UInt64, snapshot: String)
    case load(name: UInt64, snapshot: String)
    case deleteSnapshot(snapshot: String)
    case listSnapshots(prefix: String)

    enum Kind: Sendable {
        case workingSet
        case matcher(Grammar)
        case frames(Data)

        var encoded: MessagePack {
            switch self {
            case .workingSet: .string("working_set")
            case .matcher(let grammar): .map(["matcher": grammar.encoded])
            case .frames(let data): .map(["frames": .binary(data)])
            }
        }
    }

    enum Grammar: Sendable {
        case jsonSchema(String)
        case regex(String)
        case ebnf(String)
        case json

        var encoded: MessagePack {
            switch self {
            case .jsonSchema(let schema): .map(["json_schema": .string(schema)])
            case .regex(let regex): .map(["regex": .string(regex)])
            case .ebnf(let ebnf): .map(["ebnf": .string(ebnf)])
            case .json: .string("json")
            }
        }
    }

    var encoded: MessagePack {
        switch self {
        case .new(let name, let kind):
            .map(["op": .string("new"), "name": .int(Int64(name)), "kind": kind.encoded])
        case .fork(let name, let from):
            .map(["op": .string("fork"), "name": .int(Int64(name)), "from": .int(Int64(from))])
        case .close(let name):
            .map(["op": .string("close"), "name": .int(Int64(name))])
        case .stat(let name):
            .map(["op": .string("stat"), "name": .int(Int64(name))])
        case .setProps(let name, let props):
            .map(["op": .string("set_props"), "name": .int(Int64(name)), "props": .string(props)])
        case .save(let name, let snapshot):
            .map(["op": .string("save"), "name": .int(Int64(name)), "snapshot": .string(snapshot)])
        case .load(let name, let snapshot):
            .map(["op": .string("load"), "name": .int(Int64(name)), "snapshot": .string(snapshot)])
        case .deleteSnapshot(let snapshot):
            .map(["op": .string("delete_snapshot"), "snapshot": .string(snapshot)])
        case .listSnapshots(let prefix):
            .map(["op": .string("list_snapshots"), "prefix": .string(prefix)])
        }
    }
}

enum ServerFrame: Equatable {
    case response(correlationID: UInt32, ok: Bool, result: String)
    case processEvent(processID: String, event: String, value: String)
    case file(processID: String, hash: String, index: Int, total: Int, data: Data, name: String?)
    case other

    init(decoding data: Data) throws {
        let message = try MessagePack.decode(data)
        switch message["type"]?.string {
        case "response":
            guard let id = message["corr_id"]?.int, let ok = message["ok"]?.bool else {
                throw PieError.malformedFrame("a response without corr_id or ok")
            }
            self = .response(correlationID: UInt32(truncatingIfNeeded: id), ok: ok, result: message["result"]?.string ?? "")
        case "process_event":
            guard let id = message["process_id"]?.string, let event = message["event"]?.string else {
                throw PieError.malformedFrame("a process event without process_id or event")
            }
            self = .processEvent(processID: id, event: event, value: message["value"]?.string ?? "")
        case "file":
            guard let id = message["process_id"]?.string, let hash = message["file_hash"]?.string,
                  let index = message["chunk_index"]?.int, let total = message["total_chunks"]?.int,
                  let data = message["chunk_data"]?.binary else {
                throw PieError.malformedFrame("a file chunk without its process, hash, index, total or data")
            }
            self = .file(processID: id, hash: hash, index: Int(index), total: Int(total), data: data, name: message["name"]?.string)
        default:
            self = .other
        }
    }
}
