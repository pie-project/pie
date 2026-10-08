package org.pieproject.client

import org.msgpack.core.MessagePack
import org.msgpack.value.Value

/** crates/client-api's frames, as far as this client speaks them. */
internal sealed interface ClientFrame {
    data object Ping : ClientFrame
    data class Launch(val inferlet: String, val input: String, val captureOutputs: Boolean) : ClientFrame
    data class Signal(val processId: String, val message: String) : ClientFrame
    data class Terminate(val processId: String) : ClientFrame

    fun encoded(correlationId: Int): ByteArray {
        val id = correlationId.toUInt().toLong()
        val fields: List<Pair<String, Any>> = when (this) {
            Ping -> listOf("type" to "ping", "corr_id" to id)
            is Launch -> listOf(
                "type" to "launch_process", "corr_id" to id, "inferlet" to inferlet,
                "input" to input, "capture_outputs" to captureOutputs,
            )
            is Signal -> listOf("type" to "signal_process", "process_id" to processId, "message" to message)
            is Terminate -> listOf("type" to "terminate_process", "corr_id" to id, "process_id" to processId)
        }
        val packer = MessagePack.newDefaultBufferPacker()
        packer.packMapHeader(fields.size)
        for ((key, value) in fields) {
            packer.packString(key)
            when (value) {
                is String -> packer.packString(value)
                is Long -> packer.packLong(value)
                is Boolean -> packer.packBoolean(value)
                else -> error("unencodable $value")
            }
        }
        return packer.toByteArray()
    }
}

internal sealed interface ServerFrame {
    data class Response(val correlationId: Int, val ok: Boolean, val result: String) : ServerFrame
    data class ProcessEvent(val processId: String, val event: String, val value: String) : ServerFrame
    data object Other : ServerFrame

    companion object {
        fun decode(data: ByteArray): ServerFrame {
            val value = MessagePack.newDefaultUnpacker(data).use { it.unpackValue() }
            if (!value.isMapValue) throw PieException.MalformedFrame("a frame that is not a map")
            val message = value.asMapValue().map().entries
                .filter { (key, _) -> key.isStringValue }
                .associate { (key, field) -> key.asStringValue().asString() to field }
            fun string(key: String): String? = message[key]?.takeIf(Value::isStringValue)?.asStringValue()?.asString()
            return when (string("type")) {
                "response" -> {
                    val id = message["corr_id"]?.takeIf(Value::isIntegerValue)?.asIntegerValue()?.toLong()
                    val ok = message["ok"]?.takeIf(Value::isBooleanValue)?.asBooleanValue()?.boolean
                    if (id == null || ok == null) throw PieException.MalformedFrame("a response without corr_id or ok")
                    Response(id.toInt(), ok, string("result") ?: "")
                }
                "process_event" -> {
                    val id = string("process_id")
                    val event = string("event")
                    if (id == null || event == null) {
                        throw PieException.MalformedFrame("a process event without process_id or event")
                    }
                    ProcessEvent(id, event, string("value") ?: "")
                }
                else -> Other
            }
        }
    }
}
