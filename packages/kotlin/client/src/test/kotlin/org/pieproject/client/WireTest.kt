package org.pieproject.client

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import org.msgpack.core.MessagePack

class WireTest {
    @Test
    fun encodesTheCorrelationIdAsAnUnsignedInteger() {
        val frame = ClientFrame.Ping.encoded(-1)
        val message = MessagePack.newDefaultUnpacker(frame).unpackValue().asMapValue().map()
            .entries.associate { (key, value) -> key.asStringValue().asString() to value }
        assertEquals("ping", message.getValue("type").asStringValue().asString())
        assertEquals(4_294_967_295L, message.getValue("corr_id").asIntegerValue().toLong())
    }

    @Test
    fun leavesTheCorrelationIdOffASignal() {
        val message = MessagePack.newDefaultUnpacker(ClientFrame.Signal("p", "hi").encoded(0)).unpackValue().asMapValue()
        assertEquals(setOf("type", "process_id", "message"), message.keySet().map { it.asStringValue().asString() }.toSet())
    }

    @Test
    fun decodesAResponse() {
        val packer = MessagePack.newDefaultBufferPacker()
        packer.packMapHeader(4)
        packer.packString("type").packString("response")
        packer.packString("corr_id").packLong(7)
        packer.packString("ok").packBoolean(true)
        packer.packString("result").packString("p1")
        assertEquals(ServerFrame.Response(7, true, "p1"), ServerFrame.decode(packer.toByteArray()))
    }

    @Test
    fun rejectsAResponseWithoutItsCorrelationId() {
        val packer = MessagePack.newDefaultBufferPacker()
        packer.packMapHeader(1)
        packer.packString("type").packString("response")
        assertFailsWith<PieException.MalformedFrame> { ServerFrame.decode(packer.toByteArray()) }
    }
}
