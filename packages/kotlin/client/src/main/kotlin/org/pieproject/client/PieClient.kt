package org.pieproject.client

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.encodeToJsonElement

/** A client of a pie server: a remote `pie serve`, or a `PieServer` in this process. */
public class PieClient private constructor(private val transport: PieTransport) : AutoCloseable {
    private val lock = Any()
    private var nextCorrelationId = 0
    private val pending = HashMap<Int, CompletableDeferred<String>>()
    private val processes = HashMap<String, Channel<PieProcess.Event>>()

    /** Events that arrived before their process's launch response. */
    private val early = HashMap<String, MutableList<ServerFrame.ProcessEvent>>()
    private var failure: Throwable? = null

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default + CoroutineExceptionHandler { _, _ -> })

    public companion object {
        /** Connects to a `pie serve` at `ws://host:port`. */
        public suspend fun connect(url: String, identity: String? = "default/kotlin"): PieClient =
            connect(WebSocketTransport(url, identity))

        /** Speaks the protocol over [transport]; returns once the server answers a ping. */
        public suspend fun connect(transport: PieTransport): PieClient {
            val client = PieClient(transport)
            transport.frames
                .onEach(client::receive)
                .onCompletion { cause ->
                    if (cause !is CancellationException) client.fail(cause ?: PieException.ConnectionClosed())
                }
                .launchIn(client.scope)
            try {
                client.ping()
            } catch (error: Throwable) {
                client.close()
                throw error
            }
            return client
        }
    }

    override fun close() {
        transport.close()
        fail(PieException.ConnectionClosed())
        scope.cancel()
    }

    public suspend fun ping() {
        request(ClientFrame.Ping)
    }

    /** Launches an installed inferlet (`name` or `name@version`); [input] is encoded as the JSON its `main` receives. */
    public suspend inline fun <reified T> launch(
        inferlet: String,
        input: T,
        captureOutputs: Boolean = true,
        json: Json = Json,
    ): PieProcess = launch(inferlet, json.encodeToJsonElement(input), captureOutputs)

    /** Launches an inferlet with [input] as its JSON input. */
    public suspend fun launch(
        inferlet: String,
        input: JsonElement = JsonObject(emptyMap()),
        captureOutputs: Boolean = true,
    ): PieProcess {
        val id = request(ClientFrame.Launch(inferlet, input.toString(), captureOutputs))
        val events = Channel<PieProcess.Event>(Channel.UNLIMITED)
        events.invokeOnClose { cause ->
            if (cause is CancellationException) scope.launch { terminate(id) }
        }
        synchronized(lock) {
            processes[id] = events
            early.remove(id)?.forEach(::deliver)
        }
        return PieProcess(id, events.consumeAsFlow(), this)
    }

    internal suspend fun signal(processId: String, message: String) {
        synchronized(lock) { failure?.let { throw it } }
        transport.send(ClientFrame.Signal(processId, message).encoded(0))
    }

    internal suspend fun terminate(processId: String) {
        request(ClientFrame.Terminate(processId))
    }

    private suspend fun request(frame: ClientFrame): String {
        val answer = CompletableDeferred<String>()
        val id = synchronized(lock) {
            failure?.let { throw it }
            nextCorrelationId += 1
            pending[nextCorrelationId] = answer
            nextCorrelationId
        }
        try {
            transport.send(frame.encoded(id))
            return answer.await()
        } finally {
            synchronized(lock) { pending.remove(id) }
        }
    }

    private fun receive(data: ByteArray) {
        val frame = try {
            ServerFrame.decode(data)
        } catch (_: Exception) {
            return
        }
        synchronized(lock) {
            when (frame) {
                is ServerFrame.Response -> pending.remove(frame.correlationId)?.let {
                    if (frame.ok) it.complete(frame.result) else it.completeExceptionally(PieException.RequestFailed(frame.result))
                }
                is ServerFrame.ProcessEvent ->
                    if (frame.processId in processes) deliver(frame) else early.getOrPut(frame.processId, ::mutableListOf).add(frame)
                ServerFrame.Other -> Unit
            }
        }
    }

    private fun deliver(frame: ServerFrame.ProcessEvent) {
        val events = processes[frame.processId] ?: return
        when (frame.event) {
            "stdout" -> events.trySend(PieProcess.Event.Stdout(frame.value))
            "stderr" -> events.trySend(PieProcess.Event.Stderr(frame.value))
            "message" -> events.trySend(PieProcess.Event.Message(frame.value))
            "return" -> {
                processes.remove(frame.processId)
                events.trySend(PieProcess.Event.Returned(frame.value))
                events.close()
            }
            "error" -> {
                processes.remove(frame.processId)
                events.close(PieException.ProcessFailed(frame.value))
            }
        }
    }

    private fun fail(error: Throwable) {
        synchronized(lock) {
            if (failure == null) failure = error
            pending.values.forEach { it.completeExceptionally(error) }
            pending.clear()
            processes.values.forEach { it.close(error) }
            processes.clear()
        }
    }
}

/**
 * A launched inferlet. [events] completes after [Event.Returned], or throws
 * [PieException.ProcessFailed]; it can be collected once, and cancelling its
 * collection terminates the inferlet.
 */
public class PieProcess internal constructor(
    public val id: String,
    public val events: Flow<Event>,
    private val client: PieClient,
) {
    public sealed interface Event {
        public data class Stdout(public val text: String) : Event
        public data class Stderr(public val text: String) : Event

        /** A `session::send` from the inferlet. */
        public data class Message(public val text: String) : Event

        /** What `main` returned; always the last event. */
        public data class Returned(public val value: String) : Event
    }

    /** Delivers [message] to the inferlet's `session::receive`. */
    public suspend fun send(message: String) {
        client.signal(id, message)
    }

    public suspend fun terminate() {
        client.terminate(id)
    }

    /** Collects [events] and returns what `main` returned. */
    public suspend fun result(): String =
        events.filterIsInstance<Event.Returned>().firstOrNull()?.value ?: throw PieException.ConnectionClosed()
}
