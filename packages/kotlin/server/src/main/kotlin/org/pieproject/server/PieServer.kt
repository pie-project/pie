package org.pieproject.server

import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.pieproject.client.PieClient
import org.pieproject.client.PieException
import org.pieproject.client.PieTransport

/**
 * pie in this process: the runtime, the Vulkan engine and the inferlet
 * sandbox. [connect] hands out clients whose frames never leave it.
 */
public class PieServer private constructor(private val handle: Long) : AutoCloseable {
    /** What booted. */
    @Serializable
    public data class Summary(
        public val model: String,
        public val sku: String,
        public val trace: String,
        @SerialName("weight_bytes") public val weightBytes: Long,
        @SerialName("kv_pages") public val kvPages: Int,
        @SerialName("kv_page_size") public val kvPageSize: Int,
        @SerialName("max_lanes") public val maxLanes: Int,
        @SerialName("max_tokens") public val maxTokens: Int,
    )

    /** The boot configuration (`runtime::embed::BootConfig`), sized for a phone. */
    @Serializable
    public data class Configuration(
        /** Share of the device's GPU memory the engine may use. */
        @SerialName("gpu_mem_utilization") public val gpuMemoryUtilization: Double = 0.6,
        @SerialName("max_total_pages") public val maxTotalPages: Int = 512,
        @SerialName("max_forward_tokens") public val maxForwardTokens: Int = 512,
        @SerialName("max_forward_requests") public val maxForwardRequests: Int = 4,
        @SerialName("max_state_slots") public val maxStateSlots: Int = 16,
        @SerialName("max_model_len") public val maxModelLength: Int = 4096,
        /** The cap on each inferlet's linear memory. */
        @SerialName("sandbox_memory_mb") public val sandboxMemoryMB: Int = 256,
        @SerialName("max_concurrent_processes") public val maxConcurrentProcesses: Int? = 4,
        /** The SKU to serve the artifact as; null reads it from the artifact. */
        public val sku: String? = null,
    )

    /**
     * A language script inferlets (`x.py`, `x.js`) run in: the component that
     * hosts them. `language-python` and `language-javascript` provide
     * `Language.python` and `Language.javascript`.
     */
    public class Language(public val name: String, private val component: () -> ByteArray) {
        internal fun read(): ByteArray = component()

        public companion object
    }

    public val summary: Summary = json.decodeFromString(NativeCore.summary(handle))

    public companion object {
        internal val json = Json {
            ignoreUnknownKeys = true
            encodeDefaults = true
            explicitNulls = false
        }

        /**
         * Boots [model] (a `.vulkan.zt` from `pie model import`) with [languages]
         * installed; [home] (such as `File(context.cacheDir, "pie")`) holds the
         * inferlet cache. One server per process.
         */
        public suspend fun start(
            model: File,
            home: File,
            configuration: Configuration = Configuration(),
            languages: List<Language> = emptyList(),
        ): PieServer = withContext(Dispatchers.IO) {
            home.mkdirs()
            val config = json.encodeToString(Configuration.serializer(), configuration)
            val server = PieServer(NativeCore.start(model.path, config, home.path))
            languages.forEach { server.install(it) }
            server
        }
    }

    public suspend fun connect(): PieClient {
        val transport = InProcessTransport(handle, NativeCore.openSession(handle))
        return PieClient.connect(transport)
    }

    /** Installs a program; [file] names it (`x.wasm`, `x.py`, `x.js`). Returns its `name@version`, which `PieClient.launch` takes. */
    public suspend fun install(program: ByteArray, file: String, version: String? = null): String =
        withContext(Dispatchers.IO) { NativeCore.install(handle, program, file, version) }

    public suspend fun install(file: File, version: String? = null): String =
        install(withContext(Dispatchers.IO) { file.readBytes() }, file.name, version)

    /** Installs [language], so script inferlets in it install and run. */
    public suspend fun install(language: Language) {
        withContext(Dispatchers.IO) { NativeCore.installLanguage(handle, language.name, language.read()) }
    }

    /**
     * Stops the runtime and releases the engine's memory; clients fail from
     * then on. The runtime boots once per process, so no server can start
     * afterwards.
     */
    public suspend fun shutdown() {
        withContext(Dispatchers.IO) { NativeCore.shutdown(handle) }
    }

    /** [shutdown], blocking. */
    override fun close() {
        NativeCore.shutdown(handle)
    }
}

/** One session, drained by a thread of its own. Closing the session wakes that thread at once. */
internal class InProcessTransport(private val handle: Long, private val session: Int) : PieTransport {
    private val incoming = Channel<ByteArray>(Channel.UNLIMITED)
    override val frames: Flow<ByteArray> = incoming.receiveAsFlow()
    private val closed = AtomicBoolean(false)

    init {
        thread(name = "pie-session-$session", isDaemon = true) {
            try {
                while (!closed.get()) {
                    NativeCore.recvFrames(handle, session, 5_000, 64).forEach { incoming.trySend(it) }
                }
                incoming.close()
            } catch (error: PieException) {
                incoming.close(if (closed.get()) null else error)
            }
        }
    }

    override suspend fun send(frame: ByteArray) {
        if (closed.get()) throw PieException.ConnectionClosed()
        NativeCore.sendFrame(handle, session, frame)
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) NativeCore.closeSession(handle, session)
    }
}
