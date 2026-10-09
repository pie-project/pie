package org.pieproject.server

import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
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
    public class Summary internal constructor(
        public val model: String,
        public val sku: String,
        public val trace: String,
        @SerialName("weight_bytes") public val weightBytes: Long,
        @SerialName("kv_pages") public val kvPages: Int,
        @SerialName("kv_page_size") public val kvPageSize: Int,
        @SerialName("max_lanes") public val maxLanes: Int,
        @SerialName("max_tokens") public val maxTokens: Int,
    )

    /** The boot configuration (`worker::embedded::Settings`), sized for a phone. */
    @Serializable
    public class Configuration internal constructor() {
        /** Share of the device's GPU memory the engine may use. */
        @SerialName("gpu_mem_utilization") public var gpuMemoryUtilization: Double = 0.6
        @SerialName("max_total_pages") public var maxTotalPages: Int = 512
        @SerialName("max_forward_tokens") public var maxForwardTokens: Int = 512
        @SerialName("max_forward_requests") public var maxForwardRequests: Int = 4
        @SerialName("max_state_slots") public var maxStateSlots: Int = 16
        @SerialName("max_model_len") public var maxModelLength: Int = 4096

        /** The cap on each inferlet's linear memory. */
        @SerialName("sandbox_memory_mb") public var sandboxMemoryMB: Int = 256
        @SerialName("max_concurrent_processes") public var maxConcurrentProcesses: Int? = 4

        /** The SKU to serve the artifact as; null reads it from the artifact. */
        public var sku: String? = null
    }

    /**
     * A language script inferlets (`x.py`, `x.js`) run in: the component that
     * hosts them. `language-python` and `language-javascript` provide
     * `Language.python` and `Language.javascript`.
     */
    public class Language(public val name: String, internal val component: () -> ByteArray) {
        public companion object
    }

    public val summary: Summary = json.decodeFromString(NativeCore.summary(handle))

    /** `host:port` the gateway listens on with `listen` (the OS's port for port 0), else null. */
    public val listenAddress: String? = NativeCore.listenAddr(handle)

    public companion object {
        internal val json = Json {
            ignoreUnknownKeys = true
            encodeDefaults = true
            explicitNulls = false
        }

        /**
         * Boots [model] (a `.vulkan.zt` from `pie model import`) with [languages]
         * installed; [home] (such as `File(context.cacheDir, "pie")`) holds the
         * inferlet cache. With [listen] (`"127.0.0.1:8080"`) pie's gateway also
         * serves it there: its WebSocket and the OpenAI-compatible HTTP routes.
         * One server per process.
         */
        public suspend fun start(
            model: File,
            home: File,
            languages: List<Language> = emptyList(),
            listen: String? = null,
            configure: Configuration.() -> Unit = {},
        ): PieServer = withContext(Dispatchers.IO) {
            home.mkdirs()
            val config = json.encodeToString(Configuration.serializer(), Configuration().apply(configure))
            val server = PieServer(NativeCore.start(model.path, config, home.path, listen))
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
        withContext(Dispatchers.IO) { NativeCore.installLanguage(handle, language.name, language.component()) }
    }

    /**
     * Stops the runtime and releases the engine's memory; clients fail from
     * then on. The runtime boots once per process, so no server can start
     * afterwards.
     */
    public suspend fun shutdown() {
        withContext(Dispatchers.IO) { NativeCore.shutdown(handle) }
    }

    override fun close() {
        NativeCore.shutdown(handle)
    }
}

internal class InProcessTransport(private val handle: Long, private val session: Int) : PieTransport {
    private val closed = AtomicBoolean(false)

    override val frames: Flow<ByteArray> = flow {
        while (!closed.get()) {
            NativeCore.recvFrames(handle, session, 5_000, 64).forEach { emit(it) }
        }
    }.catch { error -> if (!closed.get()) throw error }.flowOn(Dispatchers.IO)

    override suspend fun send(frame: ByteArray) {
        if (closed.get()) throw PieException.ConnectionClosed()
        NativeCore.sendFrame(handle, session, frame)
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) NativeCore.closeSession(handle, session)
    }
}
