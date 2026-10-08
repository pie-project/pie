# Pie for Kotlin

Gradle modules, group `org.pieproject`:

- **`client`** (JVM) speaks pie's client protocol (MessagePack frames,
  crates/client-api). It talks to a remote `pie serve` over WebSocket, or to
  the runtime a `PieServer` runs in this process.
- **`server`** (Android, arm64) runs pie inside an app: the runtime, the
  Vulkan engine and the inferlet sandbox in-process, through
  `runtime::embed::Server` (what the Swift package wraps too).
- **`language-python`**, **`language-javascript`** add the components script
  inferlets (`x.py`, `x.js`) run in.

```kotlin
import org.pieproject.client.PieProcess
import org.pieproject.language.python
import org.pieproject.server.PieServer

@Serializable data class Prompt(val prompt: String)

val server = PieServer.start(
    model = File(context.filesDir, "qwen.vulkan.zt"),
    home = File(context.cacheDir, "pie"),
    languages = listOf(PieServer.Language.python),
) { maxModelLength = 8192 }                       // PieServer.Configuration, sized for a phone by default
val client = server.connect()                     // frames never leave the process
val name = server.install(File(context.filesDir, "main.py"))
client.launch(name, Prompt("Hi!")).events.collect { event ->
    when (event) {
        is PieProcess.Event.Stdout -> print(event.text)
        is PieProcess.Event.Message -> print(event.text)
        is PieProcess.Event.Returned -> println("returned ${event.value}")
        is PieProcess.Event.Stderr -> Unit
    }
}
server.shutdown()
```

`events` completes after `Returned` and throws `PieException.ProcessFailed`
when the inferlet fails; cancelling its collection terminates the inferlet.
The built-in inferlets (`compat-openai`, ...) are registered at boot.
`PieClient.connect("ws://host:port")` reaches a `pie serve` with the same API.

## Models

`PieServer` takes a `.vulkan.zt` artifact from a Vulkan build of pie
(`pie model import ... --sku <sku>`). The engine needs a GPU with Vulkan 1.1,
`shaderInt16` and 16-bit storage buffers.

## Build

```bash
./build-native.sh                 # core/ → server/src/main/jniLibs/arm64-v8a (NDK, cargo-ndk)
../../scripts/build-languages.sh  # the language components → language-*/src/main/resources
./gradlew test assembleRelease    # the AARs and the client jar
```

`core/` is the Rust core (`pie-kotlin-core`, a workspace member). On a Mac it
builds against Metal, so the JVM tests can boot it end to end:

```bash
cargo build --release -p pie-kotlin-core
PIE_NATIVE_DIR=$PWD/../../target/release PIE_MODEL=<model.metal.zt> ./gradlew :server:testDebugUnitTest
```

## Limits

- Not yet run on an Android device: the core cross-compiles and the Kotlin
  layer runs end to end on a Mac, but the Vulkan engine is unproven on mobile
  GPUs.
- The runtime boots once per process; after `shutdown()` no server can start.
- `libpie_server.so` is about 51 MB (18 MB compressed); the Python component
  adds about 38 MB (14 MB compressed).
