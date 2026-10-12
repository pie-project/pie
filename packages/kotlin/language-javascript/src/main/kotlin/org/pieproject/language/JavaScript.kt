package org.pieproject.language

import org.pieproject.client.PieException
import org.pieproject.server.PieServer

/** JavaScript inferlets: StarlingMonkey and `@pie-project/inferlet` in one component. */
public val PieServer.Language.Companion.javascript: PieServer.Language by lazy {
    PieServer.Language("javascript") {
        val resource = PieServer::class.java.getResourceAsStream("/org/pieproject/language/javascript.wasm")
            ?: throw PieException.Server("javascript.wasm was not bundled; scripts/build-languages.sh builds it")
        resource.use { it.readBytes() }
    }
}
