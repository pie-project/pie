import Foundation
import Testing
@testable import PieClient

private func kind(_ message: MessagePack) -> String? { message["type"]?.string }

/// The one op of a `handles` frame.
private func op(_ message: MessagePack) -> MessagePack? {
    if case .array(let ops)? = message["ops"], ops.count == 1 { return ops[0] }
    return nil
}

private func ops(_ mock: MockServer) -> [MessagePack] { mock.sent("handles").compactMap(op) }

/// Answers each `handles` frame with `answer` of its one op, as the JSON text
/// the server writes.
private func handles(_ answer: @escaping @Sendable (MessagePack) -> String = { _ in "{\"ok\":true}" }) -> MockServer {
    MockServer { message, server in
        switch kind(message) {
        case "ping": server.respond(to: message)
        case "handles": server.respond(to: message, result: "[\(answer(op(message) ?? .null))]")
        default: break
        }
    }
}

/// Refuses every frame but a ping with `reason`.
private func refusing(_ reason: String) -> MockServer {
    MockServer { message, server in
        if kind(message) == "ping" {
            server.respond(to: message)
        } else {
            server.respond(to: message, ok: false, result: reason)
        }
    }
}

private func chunk(_ reader: String, _ index: Int, _ total: Int, _ part: String, hash: String = "h") -> [String: MessagePack] {
    ["type": .string("file"), "process_id": .string(reader), "file_hash": .string(hash), "chunk_index": .int(Int64(index)),
     "total_chunks": .int(Int64(total)), "chunk_data": .binary(Data(part.utf8))]
}

/// Answers each read with `parts` as a file's chunks under its reader, then a
/// `return`, as the server does: in its own lower-case form of the id.
private func reads(_ parts: String...) -> MockServer {
    MockServer { message, server in
        switch kind(message) {
        case "ping": server.respond(to: message)
        case "read_handle":
            let reader = (message["reader"]?.string ?? "").lowercased()
            server.respond(to: message, result: reader)
            for (index, part) in parts.enumerated() { server.push(chunk(reader, index, parts.count, part)) }
            server.event(reader, "return", "")
        default: break
        }
    }
}

struct HandleTests {
    @Test func newHandlesTakeSuccessiveNames() async throws {
        let mock = handles()
        let client = try await PieClient(transport: mock)
        #expect(try await client.workingSet().name == 1)
        #expect(try await client.frames().name == 2)
        #expect(await client.pcm().name == 3)
        #expect(try await client.matcher().name == 4)
        let sent = ops(mock)
        #expect(sent[0]["op"]?.string == "new")
        #expect(sent[0]["kind"]?.string == "working_set")
        #expect(sent[1]["name"]?.int == 4)
        #expect(sent[1]["kind"]?["matcher"]?.string == "json")
    }

    @Test func aMatcherNamesItsGrammar() async throws {
        let mock = handles()
        let client = try await PieClient(transport: mock)
        _ = try await client.matcher(jsonSchema: "{}")
        _ = try await client.matcher(regex: "a+")
        _ = try await client.matcher(ebnf: "root ::= \"a\"")
        let grammars = ops(mock).map { $0["kind"]?["matcher"] }
        #expect(grammars[0]?["json_schema"]?.string == "{}")
        #expect(grammars[1]?["regex"]?.string == "a+")
        #expect(grammars[2]?["ebnf"]?.string == "root ::= \"a\"")
    }

    @Test func framesUploadAsBinary() async throws {
        let mock = handles()
        let client = try await PieClient(transport: mock)
        _ = try await client.frames(Data([1, 2, 3]))
        #expect(ops(mock).first?["kind"]?["frames"]?.binary == Data([1, 2, 3]))
    }

    @Test func aForkKeepsItsTypeAndNamesItsOrigin() async throws {
        let mock = handles()
        let client = try await PieClient(transport: mock)
        let workingSet = try await client.workingSet()
        let fork: WorkingSet = try await workingSet.fork()
        #expect(fork.name == 2)
        let op = ops(mock)[1]
        #expect(op["op"]?.string == "fork")
        #expect(op["name"]?.int == 2)
        #expect(op["from"]?.int == 1)
    }

    @Test func statAnswersWhatTheServerSaid() async throws {
        let mock = handles { op in op["op"]?.string == "stat" ? "{\"state\":\"parked\",\"tokens\":12}" : "{\"ok\":true}" }
        let client = try await PieClient(transport: mock)
        let stat = try await client.workingSet().stat()
        let fields = try JSONSerialization.jsonObject(with: stat) as? [String: Any]
        #expect(fields?["tokens"] as? Int == 12)
        #expect(fields?["state"] as? String == "parked")
    }

    @Test func setPropsSendsAnObjectWhoseNullsRemove() async throws {
        let mock = handles()
        let client = try await PieClient(transport: mock)
        let props: [String: Int?] = ["priority": 3, "deadline": nil]
        try await client.workingSet().setProps(props)
        let op = try #require(ops(mock).last)
        #expect(op["op"]?.string == "set_props")
        #expect(op["name"]?.int == 1)
        let text = try #require(op["props"]?.string)
        let sent = try #require(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        #expect(sent["priority"] as? Int == 3)
        #expect(sent["deadline"] is NSNull)
    }

    @Test func closeNamesTheHandle() async throws {
        let mock = handles()
        let client = try await PieClient(transport: mock)
        try await client.workingSet().close()
        #expect(ops(mock).last?["op"]?.string == "close")
        #expect(ops(mock).last?["name"]?.int == 1)
    }

    @Test(arguments: [
        ("unknown", HandleErrorCode.unknown), ("dead", .dead), ("quota", .quota), ("in_use", .inUse), ("invalid", .invalid),
    ])
    func aRefusedOpCarriesItsCode(wire: String, code: HandleErrorCode) async throws {
        let mock = handles { _ in "{\"error\":\"\(wire): no\",\"code\":\"\(wire)\"}" }
        let client = try await PieClient(transport: mock)
        await #expect(throws: PieError.handleFailed(code, "\(wire): no")) { try await client.workingSet() }
    }

    @Test func aRefusalWithoutACodeFieldReadsItsText() async throws {
        let client = try await PieClient(transport: handles { _ in "{\"error\":\"quota: full\"}" })
        await #expect(throws: PieError.handleFailed(.quota, "quota: full")) { try await client.workingSet() }
    }

    @Test func aWholeFrameRefusedStillReadsAsAHandleError() async throws {
        let reason = "dead: the worker holding this session's handles is gone"
        let client = try await PieClient(transport: refusing(reason))
        await #expect(throws: PieError.handleFailed(.dead, reason)) { try await client.workingSet() }
    }

    @Test func aRefusalNamingNoCodeHasNone() async throws {
        let client = try await PieClient(transport: refusing("bad client frame"))
        await #expect(throws: PieError.handleFailed(nil, "bad client frame")) { try await client.workingSet() }
    }

    @Test func launchMovesHandlesInAndNamesWhatComesBack() async throws {
        let mock = MockServer { message, server in
            switch kind(message) {
            case "ping": server.respond(to: message)
            case "handles": server.respond(to: message, result: "[{\"ok\":true}]")
            case "launch_process":
                server.respond(to: message, result: "p1")
                server.event("p1", "return", "ok")
            default: break
            }
        }
        let client = try await PieClient(transport: mock)
        let workingSet = try await client.workingSet()
        let picture = try await client.frames()
        let process = try await client.launch(
            "draw",
            input: ["ws": workingSet.marker("ws")],
            handles: ["ws": workingSet],
            returns: ["ws": workingSet, "picture": picture]
        )
        #expect(try await process.result() == "ok")
        let launch = try #require(mock.sent("launch_process").first)
        #expect(launch["handles"]?["ws"]?.int == 1)
        #expect(launch["returns"]?["ws"]?.int == 1)
        #expect(launch["returns"]?["picture"]?.int == 2)
        #expect(launch["input"]?.string == "{\"ws\":{\"$handle\":\"ws\"}}")
    }

    @Test func aLaunchWithoutHandlesWritesThemEmpty() async throws {
        let mock = server { message, server in server.respond(to: message, result: "p1") }
        let client = try await PieClient(transport: mock)
        _ = try await client.launch("echo")
        let launch = try #require(mock.sent("launch_process").first)
        #expect(launch["handles"] == .map([:]))
        #expect(launch["returns"] == .map([:]))
    }

    @Test func aLaunchRefusedOverAHandleCarriesItsCode() async throws {
        let mock = server { message, server in server.respond(to: message, ok: false, result: "in_use: handle 1 is lent") }
        let client = try await PieClient(transport: mock)
        await #expect(throws: PieError.handleFailed(.inUse, "in_use: handle 1 is lent")) { try await client.launch("echo") }
    }

    @Test func markersNameTheKindAKindNeeds() async throws {
        let client = try await PieClient(transport: handles())
        #expect(try await client.workingSet().marker("a") == ["$handle": "a"])
        #expect(try await client.frames().marker("a") == ["$handle": "a", "kind": "frames"])
        #expect(await client.pcm().marker("a") == ["$handle": "a", "kind": "pcm"])
        #expect(try await client.matcher().marker("a") == ["$handle": "a", "kind": "matcher"])
    }

    @Test func readReassemblesAMediaHandle() async throws {
        let mock = reads("PN", "G!")
        let client = try await PieClient(transport: mock)
        let bytes = try await client.frames().read(format: "png")
        #expect(bytes == Data("PNG!".utf8))
        let read = try #require(mock.sent("read_handle").first)
        #expect(read["name"]?.int == 1)
        #expect(read["format"]?.string == "png")
    }

    @Test func readingAnEmptyHandleIsEmpty() async throws {
        let client = try await PieClient(transport: reads())
        #expect(try await client.pcm().read(format: "wav").isEmpty)
    }

    @Test func twoReadsOfOneHandleKeepTheirChunksApart() async throws {
        let mock = MockServer { message, server in
            switch kind(message) {
            case "ping": server.respond(to: message)
            case "read_handle":
                server.respond(to: message, result: message["reader"]?.string ?? "")
                let readers = server.sent("read_handle").compactMap { $0["reader"]?.string }
                guard readers.count == 2, message["reader"]?.string == readers.last else { return }
                // Both are part-way when the second part of either lands.
                for reader in readers { server.push(chunk(reader, 0, 2, "PN")) }
                for reader in readers {
                    server.push(chunk(reader, 1, 2, "G!"))
                    server.event(reader, "return", "")
                }
            default: break
            }
        }
        let client = try await PieClient(transport: mock)
        let handle = try await client.frames()
        async let one = handle.read(format: "png")
        async let two = handle.read(format: "png")
        #expect(try await one == Data("PNG!".utf8))
        #expect(try await two == Data("PNG!".utf8))
    }

    @Test func aRefusedReadCarriesItsCode() async throws {
        let client = try await PieClient(transport: refusing("unknown: no handle 9"))
        await #expect(throws: PieError.handleFailed(.unknown, "unknown: no handle 9")) {
            try await client.pcm().read(format: "wav")
        }
    }

    @Test func aReadThatFailsPartWayCarriesItsCode() async throws {
        let mock = MockServer { message, server in
            switch kind(message) {
            case "ping": server.respond(to: message)
            case "read_handle":
                let reader = message["reader"]?.string ?? ""
                server.respond(to: message, result: reader)
                server.event(reader, "error", "invalid: no such format")
            default: break
            }
        }
        let client = try await PieClient(transport: mock)
        await #expect(throws: PieError.handleFailed(.invalid, "invalid: no such format")) {
            try await client.pcm().read(format: "bmp")
        }
        #expect(mock.sent("terminate_process").isEmpty)
    }

    @Test func aReadFailsWithTheConnection() async throws {
        let mock = MockServer { message, server in
            if kind(message) == "ping" { server.respond(to: message) } else { server.close() }
        }
        let client = try await PieClient(transport: mock)
        await #expect(throws: PieError.connectionClosed) { try await client.pcm().read(format: "wav") }
    }

    @Test func aCancelledReadThrowsAndTerminatesNothing() async throws {
        let mock = MockServer { message, server in
            switch kind(message) {
            case "ping": server.respond(to: message)
            case "read_handle": server.respond(to: message, result: message["reader"]?.string ?? "")
            default: break
            }
        }
        let client = try await PieClient(transport: mock)
        let handle = await client.pcm()
        let read = Task { try await handle.read(format: "wav") }
        try await Task.sleep(for: .milliseconds(50))
        read.cancel()
        await #expect(throws: CancellationError.self) { try await read.value }
        #expect(mock.sent("terminate_process").isEmpty)
    }

    @Test func aFileFromAProcessIsAnEvent() async throws {
        let mock = server { message, server in
            server.push(chunk("p1", 0, 1, "PNG!").merging(["name": .string("preview-0.png")]) { $1 })
            server.event("p1", "return", "ok")
            server.respond(to: message, result: "p1")
        }
        let client = try await PieClient(transport: mock)
        var events: [PieProcess.Event] = []
        for try await event in try await client.launch("draw").events { events.append(event) }
        #expect(events == [.file(name: "preview-0.png", data: Data("PNG!".utf8)), .returned("ok")])
    }

    @Test func saveSaysWhatItKept() async throws {
        let mock = handles { op in op["op"]?.string == "save" ? "{\"ok\":true,\"tokens\":12,\"bytes\":4096}" : "{\"ok\":true}" }
        let client = try await PieClient(transport: mock)
        let snapshot = try await client.workingSet().save("s42/turn7")
        #expect(snapshot == Snapshot(name: "s42/turn7", tokens: 12, bytes: 4096))
        let op = try #require(ops(mock).last)
        #expect(op["op"]?.string == "save")
        #expect(op["name"]?.int == 1)
        #expect(op["snapshot"]?.string == "s42/turn7")
    }

    @Test func loadTakesANewName() async throws {
        let mock = handles()
        let client = try await PieClient(transport: mock)
        _ = try await client.workingSet()
        let loaded = try await client.loadSnapshot("s42/turn7")
        #expect(loaded.name == 2)
        let op = try #require(ops(mock).last)
        #expect(op["op"]?.string == "load")
        #expect(op["name"]?.int == 2)
        #expect(op["snapshot"]?.string == "s42/turn7")
    }

    @Test func deleteAndListNameNoHandle() async throws {
        let mock = handles { op in
            op["op"]?.string == "list_snapshots"
                ? "{\"snapshots\":[{\"name\":\"s42/turn7\",\"tokens\":12,\"bytes\":4096}]}"
                : "{\"ok\":true}"
        }
        let client = try await PieClient(transport: mock)
        try await client.deleteSnapshot("s42/turn6")
        #expect(try await client.listSnapshots(prefix: "s42/") == [Snapshot(name: "s42/turn7", tokens: 12, bytes: 4096)])
        let sent = ops(mock)
        #expect(sent[0]["op"]?.string == "delete_snapshot")
        #expect(sent[0]["snapshot"]?.string == "s42/turn6")
        #expect(sent[0]["name"] == nil)
        #expect(sent[1]["op"]?.string == "list_snapshots")
        #expect(sent[1]["prefix"]?.string == "s42/")
    }

    @Test func aSnapshotRefusalCarriesItsCode() async throws {
        let client = try await PieClient(transport: handles { _ in "{\"error\":\"invalid: none\",\"code\":\"invalid\"}" })
        await #expect(throws: PieError.handleFailed(.invalid, "invalid: none")) { try await client.loadSnapshot("x") }
    }

    @Test func aListThatDoesNotParseIsMalformed() async throws {
        let client = try await PieClient(transport: handles { _ in "{\"snapshots\":[{\"name\":\"a\"}]}" })
        do {
            _ = try await client.listSnapshots()
            Issue.record("a list without its fields parsed")
        } catch PieError.malformedFrame {
        }
    }

    @Test func anAnswerThatDoesNotParseIsMalformed() async throws {
        let mock = MockServer { message, server in
            server.respond(to: message, result: kind(message) == "ping" ? "" : "not json")
        }
        let client = try await PieClient(transport: mock)
        do {
            _ = try await client.workingSet()
            Issue.record("an answer that is not JSON parsed")
        } catch PieError.malformedFrame {
        }
    }
}
