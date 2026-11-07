package org.pieproject.client

import org.msgpack.core.MessagePack
import org.msgpack.value.Value
import org.msgpack.value.ValueFactory

internal sealed interface ClientFrame {
    data object Ping : ClientFrame
    data class Launch(
        val inferlet: String,
        val input: String,
        val captureOutputs: Boolean,
        val handles: Map<String, Long>,
        val returns: Map<String, Long>,
    ) : ClientFrame
    data class Signal(val processId: String, val message: String) : ClientFrame
    data class Terminate(val processId: String) : ClientFrame
    data class Handles(val op: HandleOp) : ClientFrame
    data class ReadHandle(val name: Long, val format: String, val reader: String) : ClientFrame

    fun encoded(correlationId: Int): ByteArray {
        val id = ValueFactory.newInteger(correlationId.toUInt().toLong())
        val fields = when (this) {
            Ping -> mapOf("type" to string("ping"), "corr_id" to id)
            is Launch -> mapOf(
                "type" to string("launch_process"), "corr_id" to id, "inferlet" to string(inferlet),
                "input" to string(input), "capture_outputs" to ValueFactory.newBoolean(captureOutputs),
                "handles" to names(handles), "returns" to names(returns),
            )
            is Signal -> mapOf(
                "type" to string("signal_process"), "process_id" to string(processId), "message" to string(message),
            )
            is Terminate -> mapOf("type" to string("terminate_process"), "corr_id" to id, "process_id" to string(processId))
            is Handles -> mapOf("type" to string("handles"), "corr_id" to id, "ops" to ValueFactory.newArray(op.encoded()))
            is ReadHandle -> mapOf(
                "type" to string("read_handle"), "corr_id" to id, "name" to integer(name),
                "format" to string(format), "reader" to string(reader),
            )
        }
        return MessagePack.newDefaultBufferPacker().apply { packValue(map(fields)) }.toByteArray()
    }

    private fun names(handles: Map<String, Long>): Value = map(handles.mapValues { (_, name) -> integer(name) })
}

/** crates/client-api's `HandleOp`. */
internal sealed interface HandleOp {
    data class New(val name: Long, val kind: Kind) : HandleOp
    data class Fork(val name: Long, val from: Long) : HandleOp
    data class Close(val name: Long) : HandleOp
    data class Stat(val name: Long) : HandleOp

    /** [props] is a JSON object's text, which `Props` reads as well as a map. */
    data class SetProps(val name: Long, val props: String) : HandleOp
    data class Save(val name: Long, val snapshot: String) : HandleOp
    data class Load(val name: Long, val snapshot: String) : HandleOp
    data class DeleteSnapshot(val snapshot: String) : HandleOp
    data class ListSnapshots(val prefix: String) : HandleOp

    sealed interface Kind {
        data object WorkingSet : Kind
        data class Matcher(val grammar: Grammar) : Kind
        class Frames(val data: ByteArray) : Kind
    }

    sealed interface Grammar {
        data class JsonSchema(val schema: String) : Grammar
        data class Regex(val regex: String) : Grammar
        data class Ebnf(val ebnf: String) : Grammar
        data object Json : Grammar
    }

    fun encoded(): Value = map(
        when (this) {
            is New -> mapOf("op" to string("new"), "name" to integer(name), "kind" to kind.encoded())
            is Fork -> mapOf("op" to string("fork"), "name" to integer(name), "from" to integer(from))
            is Close -> mapOf("op" to string("close"), "name" to integer(name))
            is Stat -> mapOf("op" to string("stat"), "name" to integer(name))
            is SetProps -> mapOf("op" to string("set_props"), "name" to integer(name), "props" to string(props))
            is Save -> mapOf("op" to string("save"), "name" to integer(name), "snapshot" to string(snapshot))
            is Load -> mapOf("op" to string("load"), "name" to integer(name), "snapshot" to string(snapshot))
            is DeleteSnapshot -> mapOf("op" to string("delete_snapshot"), "snapshot" to string(snapshot))
            is ListSnapshots -> mapOf("op" to string("list_snapshots"), "prefix" to string(prefix))
        },
    )

    private fun Kind.encoded(): Value = when (this) {
        Kind.WorkingSet -> string("working_set")
        is Kind.Matcher -> map(mapOf("matcher" to grammar.encoded()))
        is Kind.Frames -> map(mapOf("frames" to ValueFactory.newBinary(data)))
    }

    private fun Grammar.encoded(): Value = when (this) {
        is Grammar.JsonSchema -> map(mapOf("json_schema" to string(schema)))
        is Grammar.Regex -> map(mapOf("regex" to string(regex)))
        is Grammar.Ebnf -> map(mapOf("ebnf" to string(ebnf)))
        Grammar.Json -> string("json")
    }
}

private fun string(value: String): Value = ValueFactory.newString(value)

private fun integer(value: Long): Value = ValueFactory.newInteger(value)

private fun map(fields: Map<String, Value>): Value = ValueFactory.newMap(fields.mapKeys { (key, _) -> string(key) })

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
