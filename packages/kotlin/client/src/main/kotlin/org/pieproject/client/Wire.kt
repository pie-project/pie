package org.pieproject.client

import org.msgpack.core.MessagePack
import org.msgpack.value.Value
import org.msgpack.value.ValueFactory

internal sealed interface ClientFrame {
    data object Ping : ClientFrame
    data class Launch(val inferlet: String, val input: String, val captureOutputs: Boolean) : ClientFrame
    data class Signal(val processId: String, val message: String) : ClientFrame
    data class Terminate(val processId: String) : ClientFrame

    fun encoded(correlationId: Int): ByteArray {
        val id = ValueFactory.newInteger(correlationId.toUInt().toLong())
        val fields = when (this) {
            Ping -> mapOf("type" to string("ping"), "corr_id" to id)
            is Launch -> mapOf(
                "type" to string("launch_process"), "corr_id" to id, "inferlet" to string(inferlet),
                "input" to string(input), "capture_outputs" to ValueFactory.newBoolean(captureOutputs),
            )
            is Signal -> mapOf(
                "type" to string("signal_process"), "process_id" to string(processId), "message" to string(message),
            )
            is Terminate -> mapOf("type" to string("terminate_process"), "corr_id" to id, "process_id" to string(processId))
        }
        val map = ValueFactory.newMap(fields.mapKeys { (key, _) -> string(key) })
        return MessagePack.newDefaultBufferPacker().apply { packValue(map) }.toByteArray()
    }

    private fun string(value: String): Value = ValueFactory.newString(value)
}

internal sealed interface ServerFrame {
    data class Response(val correlationId: Int, val ok: Boolean, val result: String) : ServerFrame
    data class ProcessEvent(val processId: String, val event: String, val value: String) : ServerFrame
    class FileChunk(
        val processId: String,
        val hash: String,
        val index: Int,
        val total: Int,
        val data: ByteArray,
        val name: String?,
    ) : ServerFrame
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
                "file" -> {
                    fun int(key: String): Int? = message[key]?.takeIf(Value::isIntegerValue)?.asIntegerValue()?.toInt()
                    val id = string("process_id")
                    val hash = string("file_hash")
                    val index = int("chunk_index")
                    val total = int("total_chunks")
                    val data = message["chunk_data"]?.takeIf(Value::isBinaryValue)?.asBinaryValue()?.asByteArray()
                    if (id == null || hash == null || index == null || total == null || data == null) {
                        throw PieException.MalformedFrame("a file chunk without its process, hash, index, total or data")
                    }
                    FileChunk(id, hash, index, total, data, string("name"))
                }
                else -> Other
            }
        }
    }
}
