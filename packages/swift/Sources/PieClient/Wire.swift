import Foundation

/// crates/client-api's frames, as far as this client speaks them.
enum ClientFrame {
    case ping
    case launch(inferlet: String, input: String, captureOutputs: Bool)
    case signal(processID: String, message: String)
    case terminate(processID: String)

    func encoded(correlationID: UInt32) -> Data {
        let id = MessagePack.int(Int64(correlationID))
        let fields: [String: MessagePack] = switch self {
        case .ping:
            ["type": .string("ping"), "corr_id": id]
        case .launch(let inferlet, let input, let captureOutputs):
            ["type": .string("launch_process"), "corr_id": id, "inferlet": .string(inferlet),
             "input": .string(input), "capture_outputs": .bool(captureOutputs)]
        case .signal(let processID, let message):
            ["type": .string("signal_process"), "process_id": .string(processID), "message": .string(message)]
        case .terminate(let processID):
            ["type": .string("terminate_process"), "corr_id": id, "process_id": .string(processID)]
        }
        return MessagePack.map(fields).encoded()
    }

    /// Whether the server answers this frame with a `response`.
    var expectsResponse: Bool {
        if case .signal = self { return false }
        return true
    }
}

enum ServerFrame: Equatable {
    case response(correlationID: UInt32, ok: Bool, result: String)
    case processEvent(processID: String, event: String, value: String)
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
        default:
            self = .other
        }
    }
}
