package org.pieproject.client

public sealed class PieException(message: String) : Exception(message) {
    /** The connection or session is closed. */
    public class ConnectionClosed : PieException("The connection is closed.")

    /** The server refused a request (an unknown inferlet, a bad input, ...). */
    public class RequestFailed(public val reason: String) : PieException("The server refused the request: $reason")

    /** The inferlet ended with an error. */
    public class ProcessFailed(public val reason: String) : PieException("The inferlet failed: $reason")

    /** The in-process server failed (boot, install, a shut-down server, ...). */
    public class Server(public val reason: String) : PieException(reason)

    /** The server sent a frame this client cannot read. */
    public class MalformedFrame(public val reason: String) : PieException("Malformed frame: $reason")
}
