package org.pieproject.server

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlinx.serialization.json.jsonObject

class ConfigurationTest {
    private fun encoded(configuration: PieServer.Configuration) =
        PieServer.json.parseToJsonElement(PieServer.json.encodeToString(PieServer.Configuration.serializer(), configuration)).jsonObject

    @Test
    fun encodesTheBootConfigKeys() {
        assertEquals(
            setOf(
                "gpu_mem_utilization", "max_total_pages", "max_forward_tokens", "max_forward_requests",
                "max_state_slots", "max_model_len", "sandbox_memory_mb", "max_concurrent_processes",
            ),
            encoded(PieServer.Configuration()).keys,
        )
    }

    @Test
    fun leavesUnsetOptionsOut() {
        val config = encoded(PieServer.Configuration().apply { maxConcurrentProcesses = null })
        assertNull(config["max_concurrent_processes"])
    }

    @Test
    fun decodesTheBootSummary() {
        val json = """{"model":"m.zt","deployment":"s","trace":"t","weight_bytes":453548700,"kv_pages":512,"kv_page_size":16,"max_lanes":4,"max_tokens":512}"""
        val summary = PieServer.json.decodeFromString(PieServer.Summary.serializer(), json)
        assertEquals(453_548_700L, summary.weightBytes)
        assertEquals(16, summary.kvPageSize)
    }
}
