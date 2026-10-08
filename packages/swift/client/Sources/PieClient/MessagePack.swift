import Foundation

/// The MessagePack subset pie's client protocol uses.
public enum MessagePack: Equatable, Sendable {
    case null
    case bool(Bool)
    case int(Int64)
    case double(Double)
    case string(String)
    case binary(Data)
    case array([MessagePack])
    case map([String: MessagePack])

    public subscript(key: String) -> MessagePack? {
        if case .map(let map) = self { return map[key] }
        return nil
    }

    public var string: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var int: Int64? {
        if case .int(let i) = self { return i }
        return nil
    }

    public var bool: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }
}

public struct MessagePackError: Error, CustomStringConvertible {
    public let description: String
}

extension MessagePack {
    public func encoded() -> Data {
        var out = Data()
        encode(into: &out)
        return out
    }

    private func encode(into out: inout Data) {
        switch self {
        case .null:
            out.append(0xc0)
        case .bool(let b):
            out.append(b ? 0xc3 : 0xc2)
        case .int(let i):
            if i >= 0 {
                let u = UInt64(i)
                if u < 0x80 { out.append(UInt8(u)) }
                else if u <= 0xff { out.append(0xcc); out.append(UInt8(u)) }
                else if u <= 0xffff { out.append(0xcd); out.appendBig(UInt16(u)) }
                else if u <= 0xffff_ffff { out.append(0xce); out.appendBig(UInt32(u)) }
                else { out.append(0xcf); out.appendBig(u) }
            } else if i >= -32 {
                out.append(UInt8(bitPattern: Int8(i)))
            } else if i >= Int64(Int8.min) {
                out.append(0xd0); out.append(UInt8(bitPattern: Int8(i)))
            } else if i >= Int64(Int16.min) {
                out.append(0xd1); out.appendBig(UInt16(bitPattern: Int16(i)))
            } else if i >= Int64(Int32.min) {
                out.append(0xd2); out.appendBig(UInt32(bitPattern: Int32(i)))
            } else {
                out.append(0xd3); out.appendBig(UInt64(bitPattern: i))
            }
        case .double(let d):
            out.append(0xcb); out.appendBig(d.bitPattern)
        case .string(let s):
            let bytes = Data(s.utf8)
            let n = bytes.count
            if n < 32 { out.append(0xa0 | UInt8(n)) }
            else if n <= 0xff { out.append(0xd9); out.append(UInt8(n)) }
            else if n <= 0xffff { out.append(0xda); out.appendBig(UInt16(n)) }
            else { out.append(0xdb); out.appendBig(UInt32(n)) }
            out.append(bytes)
        case .binary(let bytes):
            let n = bytes.count
            if n <= 0xff { out.append(0xc4); out.append(UInt8(n)) }
            else if n <= 0xffff { out.append(0xc5); out.appendBig(UInt16(n)) }
            else { out.append(0xc6); out.appendBig(UInt32(n)) }
            out.append(bytes)
        case .array(let items):
            let n = items.count
            if n < 16 { out.append(0x90 | UInt8(n)) }
            else if n <= 0xffff { out.append(0xdc); out.appendBig(UInt16(n)) }
            else { out.append(0xdd); out.appendBig(UInt32(n)) }
            for item in items { item.encode(into: &out) }
        case .map(let map):
            let n = map.count
            if n < 16 { out.append(0x80 | UInt8(n)) }
            else if n <= 0xffff { out.append(0xde); out.appendBig(UInt16(n)) }
            else { out.append(0xdf); out.appendBig(UInt32(n)) }
            for (key, value) in map {
                MessagePack.string(key).encode(into: &out)
                value.encode(into: &out)
            }
        }
    }

    public static func decode(_ data: Data) throws -> MessagePack {
        var reader = Reader(bytes: [UInt8](data))
        return try reader.value()
    }

    private struct Reader {
        let bytes: [UInt8]
        var at = 0

        mutating func byte() throws -> UInt8 {
            guard at < bytes.count else { throw MessagePackError(description: "truncated frame") }
            defer { at += 1 }
            return bytes[at]
        }

        mutating func take(_ n: Int) throws -> ArraySlice<UInt8> {
            guard n >= 0, at + n <= bytes.count else { throw MessagePackError(description: "truncated frame") }
            defer { at += n }
            return bytes[at..<at + n]
        }

        mutating func big(_ n: Int) throws -> UInt64 {
            try take(n).reduce(0) { $0 << 8 | UInt64($1) }
        }

        mutating func string(_ n: Int) throws -> String {
            guard let s = String(bytes: try take(n), encoding: .utf8) else {
                throw MessagePackError(description: "string is not UTF-8")
            }
            return s
        }

        mutating func array(_ n: Int) throws -> MessagePack {
            var items: [MessagePack] = []
            items.reserveCapacity(n)
            for _ in 0..<n { items.append(try value()) }
            return .array(items)
        }

        mutating func map(_ n: Int) throws -> MessagePack {
            var map: [String: MessagePack] = [:]
            for _ in 0..<n {
                let key = try value()
                let value = try value()
                switch key {
                case .string(let s): map[s] = value
                case .int(let i): map[String(i)] = value
                default: throw MessagePackError(description: "map key is neither a string nor an integer")
                }
            }
            return .map(map)
        }

        mutating func value() throws -> MessagePack {
            let tag = try byte()
            switch tag {
            case 0x00...0x7f: return .int(Int64(tag))
            case 0x80...0x8f: return try map(Int(tag & 0x0f))
            case 0x90...0x9f: return try array(Int(tag & 0x0f))
            case 0xa0...0xbf: return .string(try string(Int(tag & 0x1f)))
            case 0xc0: return .null
            case 0xc2: return .bool(false)
            case 0xc3: return .bool(true)
            case 0xc4: return .binary(Data(try take(Int(try big(1)))))
            case 0xc5: return .binary(Data(try take(Int(try big(2)))))
            case 0xc6: return .binary(Data(try take(Int(try big(4)))))
            case 0xca: return .double(Double(Float(bitPattern: UInt32(try big(4)))))
            case 0xcb: return .double(Double(bitPattern: try big(8)))
            case 0xcc: return .int(Int64(try big(1)))
            case 0xcd: return .int(Int64(try big(2)))
            case 0xce: return .int(Int64(try big(4)))
            case 0xcf: return .int(Int64(bitPattern: try big(8)))
            case 0xd0: return .int(Int64(Int8(bitPattern: UInt8(try big(1)))))
            case 0xd1: return .int(Int64(Int16(bitPattern: UInt16(try big(2)))))
            case 0xd2: return .int(Int64(Int32(bitPattern: UInt32(try big(4)))))
            case 0xd3: return .int(Int64(bitPattern: try big(8)))
            case 0xd9: return .string(try string(Int(try big(1))))
            case 0xda: return .string(try string(Int(try big(2))))
            case 0xdb: return .string(try string(Int(try big(4))))
            case 0xdc: return try array(Int(try big(2)))
            case 0xdd: return try array(Int(try big(4)))
            case 0xde: return try map(Int(try big(2)))
            case 0xdf: return try map(Int(try big(4)))
            case 0xe0...0xff: return .int(Int64(Int8(bitPattern: tag)))
            default: throw MessagePackError(description: String(format: "unsupported tag 0x%02x", tag))
            }
        }
    }
}

private extension Data {
    mutating func appendBig<T: FixedWidthInteger>(_ value: T) {
        Swift.withUnsafeBytes(of: value.bigEndian) { append(contentsOf: $0) }
    }
}
