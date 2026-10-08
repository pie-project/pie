package org.pieproject.server

import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.pieproject.language.python

/** The core end to end on the host: `PIE_NATIVE_DIR` and `PIE_MODEL` (see build.gradle.kts). */
class ServerTest {
    @Serializable
    data class Prompt(val prompt: String, val max_tokens: Int)

    @Test
    fun runsAPythonInferletInProcess(): Unit = runBlocking {
        val model = System.getenv("PIE_MODEL").orEmpty()
        assumeTrue(model.isNotEmpty(), "PIE_MODEL is not set")
        val quickstart = File("../../../examples/quickstart-py/main.py")
        val server = PieServer.start(
            model = File(model),
            home = Files.createTempDirectory("pie").toFile(),
            languages = listOf(PieServer.Language.python),
        )
        try {
            val name = server.install(quickstart.readBytes(), "quickstart.py")
            val client = server.connect()
            val result = client.launch(name, Prompt("The capital of France is", 8)).result()
            val text = Json.parseToJsonElement(result).jsonObject.getValue("text").jsonPrimitive.content
            println("${server.summary.sku}: $text")
            assertTrue(text.isNotBlank())
            client.close()
        } finally {
            server.shutdown()
        }
    }
}
