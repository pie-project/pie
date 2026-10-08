import Foundation
import Testing
@testable import PieServer

struct ConfigurationTests {
    @Test func encodesTheBootConfigKeys() throws {
        let configuration = PieServer.Configuration()
        let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(configuration)) as? [String: Any]
        #expect(Set(object?.keys ?? [:].keys) == [
            "gpu_mem_utilization", "max_total_pages", "max_forward_tokens", "max_forward_requests",
            "max_state_slots", "max_model_len", "sandbox_memory_mb", "max_concurrent_processes",
        ])
    }

    @Test func leavesUnsetOptionsOut() throws {
        var configuration = PieServer.Configuration()
        configuration.maxConcurrentProcesses = nil
        let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(configuration)) as? [String: Any]
        #expect(object?["max_concurrent_processes"] == nil)
    }

    @Test func decodesTheBootSummary() throws {
        let json = #"{"model":"m.zt","deployment":"s","trace":"t","weight_bytes":453548700,"kv_pages":512,"kv_page_size":16,"max_lanes":4,"max_tokens":512}"#
        let summary = try JSONDecoder.snakeCase.decode(PieServer.Summary.self, from: Data(json.utf8))
        #expect(summary.weightBytes == 453_548_700)
        #expect(summary.kvPageSize == 16)
    }
}
