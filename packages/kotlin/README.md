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

## Handles

A handle is state the client owns between processes: a working set, a matcher, or media. A launch moves it in, and it cannot be used until it comes back, so fork first to keep one.

```kotlin
val turn = client.workingSet()
val picture = client.frames()                      // client.frames(bytes) uploads encoded pictures
client.launch(
    name,
    buildJsonObject { put("ws", turn.marker("ws")) },
    handles = mapOf("ws" to turn),
    returns = mapOf("ws" to turn, "picture" to picture),
).result()

val png = picture.read("png")
turn.setProps(buildJsonObject { put("priority", 3); put("deadline", JsonNull) })  // JsonNull removes a prop
val branch = turn.fork()
val shape = client.matcher(jsonSchema = schema)    // or regex =, ebnf =, or none for any JSON
turn.stat()
branch.close()

val kept = turn.save("s42/turn7")                  // tokens and bytes kept
val again = client.loadSnapshot("s42/turn7")
client.listSnapshots("s42/")
client.deleteSnapshot("s42/turn7")
```

A refusal throws `PieException.HandleFailed`, whose `code` is a `HandleErrorCode`: `UNKNOWN`, `DEAD`, `QUOTA`, `IN_USE`, or `INVALID`, or null when the server named none. Props merge for the worker policy to read while the handle rests; sent while a process has it, they land with it.

## Build

```bash
./build-native.sh                 # core/ → server/src/main/jniLibs/arm64-v8a
../../scripts/build-languages.sh  # language components
./gradlew test assembleRelease
```
