import Foundation
import Testing
@testable import PieClient

struct MessagePackTests {
    @Test(arguments: [
        MessagePack.null, .bool(true), .bool(false),
        .int(0), .int(127), .int(128), .int(65_535), .int(65_536), .int(Int64(UInt32.max) + 1),
        .int(-1), .int(-32), .int(-33), .int(-129), .int(-32_769), .int(Int64.min),
        .double(1.5), .string(""), .string(String(repeating: "é", count: 40)), .string(String(repeating: "x", count: 70_000)),
        .binary(Data([0, 1, 2])), .binary(Data(count: 300)),
        .array((0..<20).map { .int(Int64($0)) }),
        .map(["type": .string("ping"), "corr_id": .int(7), "nested": .map(["ok": .bool(true)])]),
    ])
    func roundTrips(_ value: MessagePack) throws {
        #expect(try MessagePack.decode(value.encoded()) == value)
    }

    @Test func encodesTheCompactForms() {
        #expect(MessagePack.int(5).encoded() == Data([0x05]))
        #expect(MessagePack.int(-1).encoded() == Data([0xff]))
        #expect(MessagePack.string("ab").encoded() == Data([0xa2, 0x61, 0x62]))
        #expect(MessagePack.map(["a": .null]).encoded() == Data([0x81, 0xa1, 0x61, 0xc0]))
    }

    @Test func rejectsATruncatedFrame() {
        #expect(throws: MessagePackError.self) { try MessagePack.decode(Data([0xa5, 0x61])) }
    }
}
