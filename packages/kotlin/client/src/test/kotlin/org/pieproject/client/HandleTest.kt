package org.pieproject.client

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.msgpack.value.Value
import org.msgpack.value.ValueFactory

private fun Value.at(key: String): Value? =
    asMapValue().map().entries.firstOrNull { (name, _) -> name.asStringValue().asString() == key }?.value

private fun Value.text(): String = asStringValue().asString()

private fun Value.number(): Long = asIntegerValue().toLong()

private fun Map<String, Value>.op(): Value = getValue("ops").asArrayValue().single()

private fun Map<String, Value>.type(): String? = get("type")?.text()

private fun MockServer.ops() = sent("handles").map { it.op() }

/** Answers each `handles` frame with [answer] of its one op, as the JSON text the server writes. */
private fun handles(answer: (Value) -> String = { """{"ok":true}""" }) = MockServer { message, server ->
    when (message.type()) {
        "ping" -> server.respond(message)
        "handles" -> server.respond(message, result = "[${answer(message.op())}]")
    }
}

/** Refuses every frame but a ping with [reason]. */
private fun refusing(reason: String) = MockServer { message, server ->
    if (message.type() == "ping") server.respond(message) else server.respond(message, ok = false, result = reason)
}

private fun chunk(reader: String, index: Int, total: Int, part: String): Array<Pair<String, Any>> = arrayOf(
    "type" to "file", "process_id" to reader, "file_hash" to "h", "chunk_index" to index.toLong(),
    "total_chunks" to total.toLong(), "chunk_data" to ValueFactory.newBinary(part.toByteArray()),
)

/** Answers each read with [parts] as a file's chunks under its reader, then a `return`. */
private fun reads(vararg parts: String) = MockServer { message, server ->
    when (message.type()) {
        "ping" -> server.respond(message)
        "read_handle" -> {
            val reader = message.getValue("reader").text()
            server.respond(message, result = reader)
            parts.forEachIndexed { index, part -> server.push(*chunk(reader, index, parts.size, part)) }
            server.event(reader, "return", "")
        }
    }
}

class HandleTest {
    @Test
    fun newHandlesTakeSuccessiveNames(): Unit = runBlocking {
        val mock = handles()
        val client = PieClient.connect(mock)
        assertEquals(1, client.workingSet().name)
        assertEquals(2, client.frames().name)
        assertEquals(3, client.pcm().name)
        assertEquals(4, client.matcher().name)
        val (workingSet, matcher) = mock.ops()
        assertEquals("new", workingSet.at("op")?.text())
        assertEquals("working_set", workingSet.at("kind")?.text())
        assertEquals(4, matcher.at("name")?.number())
        assertEquals("json", matcher.at("kind")?.at("matcher")?.text())
    }

    @Test
    fun aMatcherNamesItsGrammar(): Unit = runBlocking {
        val mock = handles()
        val client = PieClient.connect(mock)
        client.matcher(jsonSchema = "{}")
        client.matcher(regex = "a+")
        client.matcher(ebnf = "root ::= \"a\"")
        val grammars = mock.ops().map { it.at("kind")?.at("matcher") }
        assertEquals("{}", grammars[0]?.at("json_schema")?.text())
        assertEquals("a+", grammars[1]?.at("regex")?.text())
        assertEquals("root ::= \"a\"", grammars[2]?.at("ebnf")?.text())
    }

    @Test
    fun framesUploadAsBinary(): Unit = runBlocking {
        val mock = handles()
        val client = PieClient.connect(mock)
        client.frames(byteArrayOf(1, 2, 3))
        val data = mock.ops().single().at("kind")?.at("frames")
        assertContentEquals(byteArrayOf(1, 2, 3), data?.asBinaryValue()?.asByteArray())
    }

    @Test
    fun aForkKeepsItsTypeAndNamesItsOrigin(): Unit = runBlocking {
        val mock = handles()
        val client = PieClient.connect(mock)
        val workingSet = client.workingSet()
        assertIs<WorkingSet>(workingSet.fork())
        assertIs<Frames>(client.frames(ByteArray(1)).fork())
        val fork = mock.ops()[1]
        assertEquals("fork", fork.at("op")?.text())
        assertEquals(2, fork.at("name")?.number())
        assertEquals(workingSet.name, fork.at("from")?.number())
    }

    @Test
    fun statAnswersWhatTheServerSaid(): Unit = runBlocking {
        val client = PieClient.connect(handles { op ->
            if (op.at("op")?.text() == "stat") """{"state":"parked","tokens":12}""" else """{"ok":true}"""
        })
        assertEquals(12, client.workingSet().stat().getValue("tokens").jsonPrimitive.int)
    }

    @Test
    fun setPropsSendsAnObjectWhoseNullsRemove(): Unit = runBlocking {
        val mock = handles()
        val client = PieClient.connect(mock)
        client.workingSet().setProps(
            buildJsonObject {
                put("priority", 3)
                put("deadline", JsonNull)
            },
        )
        val op = mock.ops().last()
        assertEquals("set_props", op.at("op")?.text())
        assertEquals(1, op.at("name")?.number())
        val props = Json.parseToJsonElement(op.at("props")!!.text()).jsonObject
        assertEquals(JsonPrimitive(3), props["priority"])
        assertEquals(JsonNull, props["deadline"])
    }

    @Test
    fun closeNamesTheHandle(): Unit = runBlocking {
        val mock = handles()
        val client = PieClient.connect(mock)
        client.workingSet().close()
        val op = mock.ops().last()
        assertEquals("close", op.at("op")?.text())
        assertEquals(1, op.at("name")?.number())
    }

    @Test
    fun aRefusedOpCarriesItsCode(): Unit = runBlocking {
        val codes = listOf(
            "unknown" to HandleErrorCode.UNKNOWN, "dead" to HandleErrorCode.DEAD, "quota" to HandleErrorCode.QUOTA,
            "in_use" to HandleErrorCode.IN_USE, "invalid" to HandleErrorCode.INVALID,
        )
        for ((wire, code) in codes) {
            val client = PieClient.connect(handles { """{"error":"$wire: no","code":"$wire"}""" })
            val error = assertFailsWith<PieException.HandleFailed> { client.workingSet() }
            assertEquals(code, error.code)
            assertEquals("$wire: no", error.reason)
        }
    }

    @Test
    fun aRefusalWithoutACodeFieldReadsItsText(): Unit = runBlocking {
        val client = PieClient.connect(handles { """{"error":"quota: full"}""" })
        assertEquals(HandleErrorCode.QUOTA, assertFailsWith<PieException.HandleFailed> { client.workingSet() }.code)
    }

    @Test
    fun aWholeFrameRefusedStillReadsAsAHandleError(): Unit = runBlocking {
        val client = PieClient.connect(refusing("dead: the worker holding this session's handles is gone"))
        assertEquals(HandleErrorCode.DEAD, assertFailsWith<PieException.HandleFailed> { client.workingSet() }.code)
    }

    @Test
    fun aRefusalNamingNoCodeHasNone(): Unit = runBlocking {
        val client = PieClient.connect(refusing("bad client frame: unknown variant"))
        assertEquals(null, assertFailsWith<PieException.HandleFailed> { client.workingSet() }.code)
    }

    @Test
    fun launchMovesHandlesInAndNamesWhatComesBack(): Unit = runBlocking {
        val mock = MockServer { message, server ->
            when (message.type()) {
                "ping" -> server.respond(message)
                "handles" -> server.respond(message, result = """[{"ok":true}]""")
                "launch_process" -> {
                    server.respond(message, result = "p1")
                    server.event("p1", "return", "ok")
                }
            }
        }
        val client = PieClient.connect(mock)
        val workingSet = client.workingSet()
        val picture = client.frames()
        val process = client.launch(
            "draw",
            buildJsonObject { put("ws", workingSet.marker("ws")) },
            handles = mapOf("ws" to workingSet),
            returns = mapOf("ws" to workingSet, "picture" to picture),
        )
        assertEquals("ok", process.result())
        val launch = mock.sent("launch_process").single()
        assertEquals(1, launch.getValue("handles").at("ws")?.number())
        assertEquals(1, launch.getValue("returns").at("ws")?.number())
        assertEquals(2, launch.getValue("returns").at("picture")?.number())
        assertEquals("""{"ws":{"${'$'}handle":"ws"}}""", launch.getValue("input").text())
    }

    @Test
    fun aTypedInputCarriesHandlesToo(): Unit = runBlocking {
        val mock = server { message, server -> server.respond(message, result = "p1") }
        val client = PieClient.connect(mock)
        client.launch("echo", Input("hi", 3), handles = mapOf("pcm" to client.pcm()))
        assertEquals(1, mock.sent("launch_process").single().getValue("handles").at("pcm")?.number())
    }

    @Test
    fun aLaunchWithoutHandlesWritesThemEmpty(): Unit = runBlocking {
        val mock = server { message, server -> server.respond(message, result = "p1") }
        val client = PieClient.connect(mock)
        client.launch("echo")
        val launch = mock.sent("launch_process").single()
        assertEquals(0, launch.getValue("handles").asMapValue().size())
        assertEquals(0, launch.getValue("returns").asMapValue().size())
    }

    @Test
    fun aLaunchRefusedOverAHandleCarriesItsCode(): Unit = runBlocking {
        val mock = server { message, server -> server.respond(message, ok = false, result = "in_use: handle 1 is lent") }
        val client = PieClient.connect(mock)
        assertEquals(HandleErrorCode.IN_USE, assertFailsWith<PieException.HandleFailed> { client.launch("echo") }.code)
    }

    @Test
    fun markersNameTheKindAKindNeeds(): Unit = runBlocking {
        val client = PieClient.connect(handles())
        assertEquals(setOf("\$handle"), client.workingSet().marker("a").keys)
        assertEquals("a", client.workingSet().marker("a")["\$handle"]?.jsonPrimitive?.content)
        assertEquals("frames", client.frames().marker("a")["kind"]?.jsonPrimitive?.content)
        assertEquals("pcm", client.pcm().marker("a")["kind"]?.jsonPrimitive?.content)
        assertEquals("matcher", client.matcher().marker("a")["kind"]?.jsonPrimitive?.content)
    }

    @Test
    fun readReassemblesAMediaHandle(): Unit = runBlocking {
        val mock = reads("PN", "G!")
        val client = PieClient.connect(mock)
        val bytes = withTimeout(1_000) { client.frames().read("png") }
        assertEquals("PNG!", bytes.decodeToString())
        val read = mock.sent("read_handle").single()
        assertEquals(1, read.getValue("name").number())
        assertEquals("png", read.getValue("format").text())
    }

    @Test
    fun readingAnEmptyHandleIsEmpty(): Unit = runBlocking {
        val client = PieClient.connect(reads())
        assertEquals(0, withTimeout(1_000) { client.pcm().read("wav") }.size)
    }

    @Test
    fun twoReadsOfOneHandleKeepTheirChunksApart(): Unit = runBlocking {
        val readers = mutableListOf<String>()
        val mock = MockServer { message, server ->
            when (message.type()) {
                "ping" -> server.respond(message)
                "read_handle" -> {
                    readers += message.getValue("reader").text()
                    server.respond(message, result = readers.last())
                    if (readers.size == 2) {
                        // Both are part-way when the second part of either lands.
                        for (reader in readers) server.push(*chunk(reader, 0, 2, "PN"))
                        for (reader in readers) {
                            server.push(*chunk(reader, 1, 2, "G!"))
                            server.event(reader, "return", "")
                        }
                    }
                }
            }
        }
        val client = PieClient.connect(mock)
        val handle = client.frames()
        val one = async { handle.read("png") }
        while (mock.sent("read_handle").isEmpty()) delay(1)
        val two = async { handle.read("png") }
        withTimeout(1_000) {
            assertEquals("PNG!", one.await().decodeToString())
            assertEquals("PNG!", two.await().decodeToString())
        }
    }

    @Test
    fun aRefusedReadCarriesItsCode(): Unit = runBlocking {
        val client = PieClient.connect(refusing("unknown: no handle 9"))
        assertEquals(HandleErrorCode.UNKNOWN, assertFailsWith<PieException.HandleFailed> { client.pcm().read("wav") }.code)
    }

    @Test
    fun aReadThatFailsPartWayCarriesItsCode(): Unit = runBlocking {
        val mock = MockServer { message, server ->
            when (message.type()) {
                "ping" -> server.respond(message)
                "read_handle" -> {
                    server.respond(message, result = message.getValue("reader").text())
                    server.event(message.getValue("reader").text(), "error", "invalid: no such format")
                }
            }
        }
        val client = PieClient.connect(mock)
        val error = assertFailsWith<PieException.HandleFailed> { client.pcm().read("bmp") }
        assertEquals(HandleErrorCode.INVALID, error.code)
        assertEquals(0, mock.sent("terminate_process").size, "a reader is no process to terminate")
    }

    @Test
    fun aReadFailsWithTheConnection(): Unit = runBlocking {
        val mock = MockServer { message, server ->
            if (message.type() == "ping") server.respond(message) else server.close()
        }
        val client = PieClient.connect(mock)
        assertFailsWith<PieException.ConnectionClosed> { client.pcm().read("wav") }
    }

    @Test
    fun saveSaysWhatItKept(): Unit = runBlocking {
        val mock = handles { op ->
            if (op.at("op")?.text() == "save") """{"ok":true,"tokens":12,"bytes":4096}""" else """{"ok":true}"""
        }
        val client = PieClient.connect(mock)
        assertEquals(Snapshot("s42/turn7", 12, 4096), client.workingSet().save("s42/turn7"))
        val op = mock.ops().last()
        assertEquals("save", op.at("op")?.text())
        assertEquals(1, op.at("name")?.number())
        assertEquals("s42/turn7", op.at("snapshot")?.text())
    }

    @Test
    fun loadTakesANewName(): Unit = runBlocking {
        val mock = handles()
        val client = PieClient.connect(mock)
        client.workingSet()
        val loaded = client.loadSnapshot("s42/turn7")
        assertEquals(2, loaded.name)
        val op = mock.ops().last()
        assertEquals("load", op.at("op")?.text())
        assertEquals(2, op.at("name")?.number())
        assertEquals("s42/turn7", op.at("snapshot")?.text())
    }

    @Test
    fun deleteAndListNameNoHandle(): Unit = runBlocking {
        val mock = handles { op ->
            if (op.at("op")?.text() == "list_snapshots") {
                """{"snapshots":[{"name":"s42/turn7","tokens":12,"bytes":4096}]}"""
            } else {
                """{"ok":true}"""
            }
        }
        val client = PieClient.connect(mock)
        client.deleteSnapshot("s42/turn6")
        assertEquals(listOf(Snapshot("s42/turn7", 12, 4096)), client.listSnapshots("s42/"))
        val (delete, list) = mock.ops()
        assertEquals("delete_snapshot", delete.at("op")?.text())
        assertEquals("s42/turn6", delete.at("snapshot")?.text())
        assertEquals(null, delete.at("name"))
        assertEquals("list_snapshots", list.at("op")?.text())
        assertEquals("s42/", list.at("prefix")?.text())
    }

    @Test
    fun aSnapshotRefusalCarriesItsCode(): Unit = runBlocking {
        val client = PieClient.connect(handles { """{"error":"invalid: no such snapshot","code":"invalid"}""" })
        assertEquals(HandleErrorCode.INVALID, assertFailsWith<PieException.HandleFailed> { client.loadSnapshot("x") }.code)
    }

    @Test
    fun aListThatDoesNotParseIsMalformed(): Unit = runBlocking {
        val client = PieClient.connect(handles { """{"snapshots":[{"name":"a"}]}""" })
        assertFailsWith<PieException.MalformedFrame> { client.listSnapshots() }
    }

    @Test
    fun anAnswerThatDoesNotParseIsMalformed(): Unit = runBlocking {
        val mock = MockServer { message, server ->
            if (message.type() == "ping") server.respond(message) else server.respond(message, result = "not json")
        }
        val client = PieClient.connect(mock)
        assertFailsWith<PieException.MalformedFrame> { client.workingSet() }
    }
}
