package org.pieproject.client

import java.net.URI
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString

/** Carries a [PieClient]'s MessagePack frames both ways. */
public interface PieTransport : AutoCloseable {
    /** Every frame the server sends, in order; completes when the transport closes. */
    public val frames: Flow<ByteArray>

    public suspend fun send(frame: ByteArray)

    override fun close()
}

/**
 * A `pie serve` gateway at `ws://host:port` (its `/v1/ws` route when the URL
 * names no path). [identity] is the `x-pie-identity` header (`tenant/user`)
 * the gateway requires of a client no edge proxy fronts.
 */
public class WebSocketTransport(
    url: String,
    identity: String? = "default/kotlin",
    client: OkHttpClient = OkHttpClient(),
) : PieTransport {
    private val incoming = Channel<ByteArray>(Channel.UNLIMITED)
    override val frames: Flow<ByteArray> = incoming.receiveAsFlow()
    private val socket: WebSocket

    init {
        val target = if (URI(url).path.trim('/').isEmpty()) url.trimEnd('/') + "/v1/ws" else url
        val request = Request.Builder().url(target)
        identity?.let { request.header("x-pie-identity", it) }
        socket = client.newWebSocket(request.build(), object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                incoming.trySend(bytes.toByteArray())
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                incoming.close()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                incoming.close(t)
            }
        })
    }

    override suspend fun send(frame: ByteArray) {
        if (!socket.send(frame.toByteString())) throw PieException.ConnectionClosed()
    }

    override fun close() {
        socket.close(1000, null)
        incoming.close()
    }
}
