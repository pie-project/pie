package org.pieproject.language

import org.pieproject.client.PieException
import org.pieproject.server.PieServer

/** Python inferlets: CPython and the `inferlet` package in one component. */
public val PieServer.Language.Companion.python: PieServer.Language by lazy {
    PieServer.Language("python") {
        val resource = PieServer::class.java.getResourceAsStream("/org/pieproject/language/python.wasm")
            ?: throw PieException.Server("python.wasm was not bundled; scripts/build-languages.sh builds it")
        resource.use { it.readBytes() }
    }
}
