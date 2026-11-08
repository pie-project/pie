package org.pieproject.client

import java.util.UUID
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
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.encodeToJsonElement
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long

/** A client of a pie server: a remote `pie serve`, or a `PieServer` in this process. */
public class PieClient private constructor(private val transport: PieTransport) : AutoCloseable {
    private val lock = Any()
    private var nextCorrelationId = 0
    private var lastHandle = 0L
    private val pending = HashMap<Int, CompletableDeferred<String>>()
    private val processes = HashMap<String, Channel<PieProcess.Event>>()

    /** Files part-way through their chunks, by process and hash: two reads of one handle share a hash. */
    private val downloads = HashMap<Pair<String, String>, Pair<ServerFrame.FileChunk, java.io.ByteArrayOutputStream>>()

    /** Events that arrived before their process's launch response. */
    private val early = HashMap<String, MutableList<ServerFrame>>()
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

    /** Launches an installed inferlet (`name` or `name@<hash>`); [input] is encoded as the JSON its `main` receives. */
    public suspend inline fun <reified T> launch(
        inferlet: String,
        input: T,
        captureOutputs: Boolean = true,
        json: Json = Json,
        handles: Map<String, Handle> = emptyMap(),
        returns: Map<String, Handle> = emptyMap(),
    ): PieProcess = launch(inferlet, json.encodeToJsonElement(input), captureOutputs, handles, returns)

    /**
     * Launches an inferlet with [input] as its JSON input. [handles] move into
     * the process, by the name its guest takes each under, and cannot be used
     * until they come back: fork first to keep one. [returns] is what the
     * process gives back, by the name it gives each under.
     */
    public suspend fun launch(
        inferlet: String,
        input: JsonElement = JsonObject(emptyMap()),
        captureOutputs: Boolean = true,
        handles: Map<String, Handle> = emptyMap(),
        returns: Map<String, Handle> = emptyMap(),
    ): PieProcess {
        val frame = ClientFrame.Launch(
            inferlet, input.toString(), captureOutputs, handles.mapValues { it.value.name }, returns.mapValues { it.value.name },
        )
        val id = try {
            request(frame)
        } catch (error: PieException.RequestFailed) {
            throw HandleErrorCode.of(error.reason)?.let { PieException.HandleFailed(it, error.reason) } ?: error
        }
        val events = Channel<PieProcess.Event>(Channel.UNLIMITED)
        events.invokeOnClose { cause ->
            if (cause is CancellationException) scope.launch { terminate(id) }
        }
        synchronized(lock) {
            processes[id] = events
            early.remove(id)?.forEach(::route)
        }
        return PieProcess(id, events.consumeAsFlow(), this)
    }

    /** A new, empty working set. */
    public suspend fun workingSet(): WorkingSet {
        val name = nextHandle()
        handleOp(HandleOp.New(name, HandleOp.Kind.WorkingSet))
        return WorkingSet(this, name)
    }

    /**
     * A matcher at the start of one grammar, or of any JSON given none. The
     * grammar compiles where a process binds it, and an error in it fails that
     * launch.
     */
    public suspend fun matcher(jsonSchema: String? = null, regex: String? = null, ebnf: String? = null): Matcher {
        val grammar = when {
            jsonSchema != null -> HandleOp.Grammar.JsonSchema(jsonSchema)
            regex != null -> HandleOp.Grammar.Regex(regex)
            ebnf != null -> HandleOp.Grammar.Ebnf(ebnf)
            else -> HandleOp.Grammar.Json
        }
        val name = nextHandle()
        handleOp(HandleOp.New(name, HandleOp.Kind.Matcher(grammar)))
        return Matcher(this, name)
    }

    /**
     * An upload, if [data] is given: encoded pictures, which decode where a
     * process binds them. Else a name for a process to give pictures to.
     */
    public suspend fun frames(data: ByteArray? = null): Frames {
        val name = nextHandle()
        if (data != null) handleOp(HandleOp.New(name, HandleOp.Kind.Frames(data)))
        return Frames(this, name)
    }

    /** A name for a process to give sound to. */
    public fun pcm(): Pcm = Pcm(this, nextHandle())

    /** A new working set as [snapshot] stands. */
    public suspend fun loadSnapshot(snapshot: String): WorkingSet {
        val name = nextHandle()
        handleOp(HandleOp.Load(name, snapshot))
        return WorkingSet(this, name)
    }

    /** Handles loaded from it stay. */
    public suspend fun deleteSnapshot(snapshot: String) {
        handleOp(HandleOp.DeleteSnapshot(snapshot))
    }

    /** Each snapshot under [prefix]. */
    public suspend fun listSnapshots(prefix: String = ""): List<Snapshot> {
        val answer = handleOp(HandleOp.ListSnapshots(prefix))
        return try {
            answer.getValue("snapshots").jsonArray.map {
                val snapshot = it.jsonObject
                snapshot(snapshot.getValue("name").jsonPrimitive.content, snapshot)
            }
        } catch (error: RuntimeException) {
            throw PieException.MalformedFrame("a snapshot list: ${error.message}")
        }
    }

    internal suspend fun save(name: Long, snapshot: String): Snapshot {
        val answer = handleOp(HandleOp.Save(name, snapshot))
        return try {
            snapshot(snapshot, answer)
        } catch (error: RuntimeException) {
            throw PieException.MalformedFrame("a save's answer: ${error.message}")
        }
    }

    private fun snapshot(name: String, answer: JsonObject) =
        Snapshot(name, answer.getValue("tokens").jsonPrimitive.long, answer.getValue("bytes").jsonPrimitive.long)

    internal fun nextHandle(): Long = synchronized(lock) { ++lastHandle }

    /** Sends [op] alone and returns its answer. */
    internal suspend fun handleOp(op: HandleOp): JsonObject {
        val result = handleRequest(ClientFrame.Handles(op))
        val answer = try {
            Json.parseToJsonElement(result) as JsonArray
        } catch (error: RuntimeException) {
            throw PieException.MalformedFrame("a handle answer: ${error.message}")
        }.firstOrNull() as? JsonObject ?: throw PieException.MalformedFrame("a handle answer that is not an object")
        val error = answer["error"] ?: return answer
        val reason = (error as? JsonPrimitive)?.contentOrNull ?: error.toString()
        val code = HandleErrorCode.named((answer["code"] as? JsonPrimitive)?.contentOrNull)
        throw PieException.HandleFailed(code ?: HandleErrorCode.of(reason), reason)
    }

    private suspend fun handleRequest(frame: ClientFrame): String = try {
        request(frame)
    } catch (error: PieException.RequestFailed) {
        throw PieException.HandleFailed(HandleErrorCode.of(error.reason), error.reason)
    }

    internal suspend fun read(name: Long, format: String): ByteArray {
        // Picked here so the channel is open before the first chunk can arrive.
        val reader = UUID.randomUUID().toString()
        val events = Channel<PieProcess.Event>(Channel.UNLIMITED)
        synchronized(lock) {
            failure?.let { throw it }
            processes[reader] = events
        }
        try {
            handleRequest(ClientFrame.ReadHandle(name, format, reader))
            var data = ByteArray(0)
            for (event in events) {
                if (event is PieProcess.Event.File) data = event.bytes
            }
            return data
        } catch (error: PieException.ProcessFailed) {
            throw PieException.HandleFailed(HandleErrorCode.of(error.reason), error.reason)
        } finally {
            synchronized(lock) {
                processes.remove(reader)
                downloads.keys.removeAll { it.first == reader }
            }
        }
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
                is ServerFrame.FileChunk ->
                    if (frame.processId in processes) collect(frame) else early.getOrPut(frame.processId, ::mutableListOf).add(frame)
                ServerFrame.Other -> Unit
            }
        }
    }

    private fun route(frame: ServerFrame) {
        when (frame) {
            is ServerFrame.ProcessEvent -> deliver(frame)
            is ServerFrame.FileChunk -> collect(frame)
            else -> Unit
        }
    }

    private fun collect(chunk: ServerFrame.FileChunk) {
        val events = processes[chunk.processId] ?: return
        val key = chunk.processId to chunk.hash
        val started = downloads[key] ?: if (chunk.index == 0) {
            (chunk to java.io.ByteArrayOutputStream()).also { downloads[key] = it }
        } else {
            return
        }
        started.second.write(chunk.data)
        if (chunk.index == chunk.total - 1) {
            downloads.remove(key)
            events.trySend(PieProcess.Event.File(started.first.name, started.second.toByteArray()))
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

        /** A file the inferlet sent (`session::send_file` with a name, `send_frames`), under the name it gave, if any. */
        public class File(public val name: String?, public val bytes: ByteArray) : Event

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
