# Pie for Kotlin

In-process Vulkan engine (`server`, Android arm64), client (`client`, JVM), and language components (`language-python`, `language-javascript`) under `org.pieproject`.

```kotlin
import org.pieproject.client.PieProcess
import org.pieproject.language.python
import org.pieproject.server.PieServer

@Serializable data class Prompt(val prompt: String)

val server = PieServer.start(
    model = File(context.filesDir, "qwen.vulkan.zt"),
    home = File(context.cacheDir, "pie"),
    languages = listOf(PieServer.Language.python),
)
val client = server.connect()
val name = server.install(File(context.filesDir, "main.py"))
client.launch(name, Prompt("Hi!")).events.collect { event ->
    when (event) {
        is PieProcess.Event.Stdout -> print(event.text)
        is PieProcess.Event.Message -> print(event.text)
        is PieProcess.Event.File -> File(context.filesDir, event.name ?: "out.bin").writeBytes(event.bytes)
        is PieProcess.Event.Returned -> println("returned ${event.value}")
        is PieProcess.Event.Stderr -> Unit
    }
}
server.shutdown()
```

Pass `listen = "127.0.0.1:8080"` to expose `/v1/ws` and the OpenAI-compatible HTTP routes, or use `PieClient.connect("ws://host:port")` against a remote `pie serve`.

## Build

```bash
./build-native.sh                 # core/ → server/src/main/jniLibs/arm64-v8a
../../scripts/build-languages.sh  # language components
./gradlew test assembleRelease
```
