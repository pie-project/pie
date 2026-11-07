package org.pieproject.client

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * State this session owns between processes. Lazy: nothing exists on the
 * server until a process binds it, so making one costs a name.
 */
public sealed class Handle(internal val client: PieClient, public val name: Long) {
    /** What the input carries where this handle goes, under the name [slot] it moves in by. */
    public open fun marker(slot: String): JsonObject = marked(slot)

    /** An independent copy of this handle as it stands. */
    public abstract suspend fun fork(): Handle

    internal suspend fun forked(): Long {
        val child = client.nextHandle()
        client.handleOp(HandleOp.Fork(child, name))
        return child
    }

    internal fun marked(slot: String, kind: String? = null): JsonObject = JsonObject(
        buildMap {
            put("\$handle", JsonPrimitive(slot))
            kind?.let { put("kind", JsonPrimitive(it)) }
        },
    )

    public suspend fun stat(): JsonObject = client.handleOp(HandleOp.Stat(name))

    /**
     * Merges [props] for the worker policy to read while this rests; a
     * `JsonNull` removes one. Sent while a process has it, they land with it.
     */
    public suspend fun setProps(props: JsonObject) {
        client.handleOp(HandleOp.SetProps(name, props.toString()))
    }

    public suspend fun close() {
        client.handleOp(HandleOp.Close(name))
    }
}

/** One sequence's model state. */
public class WorkingSet internal constructor(client: PieClient, name: Long) : Handle(client, name) {
    override suspend fun fork(): WorkingSet = WorkingSet(client, forked())

    /**
     * Keeps this working set at rest under [snapshot], past this session and
     * this server, and says what it kept. It stays as it was.
     */
    public suspend fun save(snapshot: String): Snapshot = client.save(name, snapshot)
}

/**
 * A grammar match in progress. It stands where it stood across processes with
 * the working set it constrains: move and fork them together.
 */
public class Matcher internal constructor(client: PieClient, name: Long) : Handle(client, name) {
    override fun marker(slot: String): JsonObject = marked(slot, "matcher")

    override suspend fun fork(): Matcher = Matcher(client, forked())
}

/**
 * Host memory a process made or a client uploaded: it moves on to another
 * process without crossing the client, and comes down only when asked.
 */
public sealed class Media(client: PieClient, name: Long) : Handle(client, name) {
    /** The whole of this handle, encoded as [format]. It stays. */
    public suspend fun read(format: String): ByteArray = client.read(name, format)
}

/** Pictures, or one: `count` frames of RGB8. */
public class Frames internal constructor(client: PieClient, name: Long) : Media(client, name) {
    override fun marker(slot: String): JsonObject = marked(slot, "frames")

    override suspend fun fork(): Frames = Frames(client, forked())
}

/** Sound: interleaved samples at a rate. */
public class Pcm internal constructor(client: PieClient, name: Long) : Media(client, name) {
    override fun marker(slot: String): JsonObject = marked(slot, "pcm")

    override suspend fun fork(): Pcm = Pcm(client, forked())
}

/** A working set kept by [WorkingSet.save]. */
public data class Snapshot(public val name: String, public val tokens: Long, public val bytes: Long)
