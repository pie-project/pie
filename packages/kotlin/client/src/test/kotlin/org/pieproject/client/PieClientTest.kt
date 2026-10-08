package org.pieproject.client

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.msgpack.core.MessagePack
import org.msgpack.value.Value

/** A server scripted per request: [script] sees each client frame and answers through [push]. */
class MockServer(private val script: (Map<String, Value>, MockServer) -> Unit) : PieTransport {
    private val incoming = Channel<ByteArray>(Channel.UNLIMITED)
    override val frames: Flow<ByteArray> = incoming.receiveAsFlow()
    private val received = mutableListOf<Map<String, Value>>()

    override suspend fun send(frame: ByteArray) {
        val message = MessagePack.newDefaultUnpacker(frame).unpackValue().asMapValue().map()
            .entries.associate { (key, value) -> key.asStringValue().asString() to value }
        synchronized(received) { received.add(message) }
        script(message, this)
    }

    override fun close() {
        incoming.close()
    }

    fun push(vararg fields: Pair<String, Any>) {
        val packer = MessagePack.newDefaultBufferPacker()
        packer.packMapHeader(fields.size)
        for ((key, value) in fields) {
            packer.packString(key)
            when (value) {
                is String -> packer.packString(value)
                is Boolean -> packer.packBoolean(value)
                is Long -> packer.packLong(value)
                is Value -> packer.packValue(value)
            }
        }
        incoming.trySend(packer.toByteArray())
    }

    fun respond(to: Map<String, Value>, ok: Boolean = true, result: String = "") =
        push("type" to "response", "corr_id" to to.getValue("corr_id"), "ok" to ok, "result" to result)

    fun event(process: String, event: String, value: String) =
        push("type" to "process_event", "process_id" to process, "event" to event, "value" to value)

    fun sent(type: String): List<Map<String, Value>> =
        synchronized(received) { received.filter { it["type"]?.asStringValue()?.asString() == type } }
}

/** Answers pings; [launch] handles each launch. */
fun server(launch: (Map<String, Value>, MockServer) -> Unit = { _, _ -> }) = MockServer { message, server ->
    when (message["type"]?.asStringValue()?.asString()) {
        "ping", "terminate_process" -> server.respond(message)
        "launch_process" -> launch(message, server)
    }
}

@Serializable
data class Input(val prompt: String, val maxTokens: Int)

class PieClientTest {
    @Test
    fun launchesWithEncodedInputAndStreamsEvents(): Unit = runBlocking {
        val mock = server { message, server ->
            server.respond(message, result = "p1")
            server.event("p1", "stdout", "hello")
            server.event("p1", "message", """{"delta":"x"}""")
            server.event("p1", "return", "done")
        }
        val client = PieClient.connect(mock)
        val process = client.launch("echo", Input(prompt = "hi", maxTokens = 3))
        assertEquals(
            listOf(PieProcess.Event.Stdout("hello"), PieProcess.Event.Message("""{"delta":"x"}"""), PieProcess.Event.Returned("done")),
            process.events.toList(),
        )
        val launch = mock.sent("launch_process").single()
        assertEquals("echo", launch.getValue("inferlet").asStringValue().asString())
        val input = Json.parseToJsonElement(launch.getValue("input").asStringValue().asString()).jsonObject
        assertEquals("hi", input.getValue("prompt").jsonPrimitive.content)
    }

    @Test
    fun keepsEventsThatBeatTheLaunchResponse(): Unit = runBlocking {
        val mock = server { message, server ->
            server.event("p2", "stdout", "early")
            server.event("p2", "return", "ok")
            server.respond(message, result = "p2")
        }
        val client = PieClient.connect(mock)
        assertEquals("ok", client.launch("echo").result())
    }

    @Test
    fun throwsTheInferletsError(): Unit = runBlocking {
        val mock = server { message, server ->
            server.respond(message, result = "p3")
            server.event("p3", "error", "boom")
        }
        val client = PieClient.connect(mock)
        val process = client.launch("echo")
        assertEquals("boom", assertFailsWith<PieException.ProcessFailed> { process.result() }.reason)
    }

    @Test
    fun throwsARefusedLaunch(): Unit = runBlocking {
        val mock = server { message, server -> server.respond(message, ok = false, result = "no such inferlet") }
        val client = PieClient.connect(mock)
        assertEquals("no such inferlet", assertFailsWith<PieException.RequestFailed> { client.launch("missing") }.reason)
    }

    @Test
    fun failsPendingRequestsWhenTheConnectionCloses(): Unit = runBlocking {
        val mock = server { _, server -> server.close() }
        val client = PieClient.connect(mock)
        assertFailsWith<PieException.ConnectionClosed> { client.launch("echo") }
        assertFailsWith<PieException.ConnectionClosed> { client.ping() }
    }

    @Test
    fun cancellingTheEventsTerminatesTheInferlet(): Unit = runBlocking {
        val mock = server { message, server ->
            server.respond(message, result = "p4")
            server.event("p4", "stdout", "working")
        }
        val client = PieClient.connect(mock)
        val process = client.launch("loop")
        val consumer = launch { process.events.collect {} }
        delay(50)
        consumer.cancel()
        withTimeout(1_000) {
            while (mock.sent("terminate_process").isEmpty()) delay(10)
        }
        assertEquals("p4", mock.sent("terminate_process").first().getValue("process_id").asStringValue().asString())
    }

    @Test
    fun refusesToConnectWithoutAPingAnswer(): Unit = runBlocking {
        val mock = MockServer { message, server -> server.respond(message, ok = false, result = "busy") }
        assertEquals("busy", assertFailsWith<PieException.RequestFailed> { PieClient.connect(mock) }.reason)
    }
}
