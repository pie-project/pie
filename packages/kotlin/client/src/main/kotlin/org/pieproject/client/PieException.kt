package org.pieproject.client

public sealed class PieException(message: String) : Exception(message) {
    public class ConnectionClosed : PieException("The connection is closed.")

    /** The server refused a request (an unknown inferlet, a bad input, ...). */
    public class RequestFailed(public val reason: String) : PieException("The server refused the request: $reason")

    /** A handle op, a launch or a read was refused; [code] says how, when the server named one. */
    public class HandleFailed(public val code: HandleErrorCode?, public val reason: String) :
        PieException("The server refused the handle: $reason")

    /** The inferlet ended with an error. */
    public class ProcessFailed(public val reason: String) : PieException("The inferlet failed: $reason")

    /** The in-process server failed (boot, install, a shut-down server, ...). */
    public class Server(public val reason: String) : PieException(reason)

    /** The server sent a frame this client cannot read. */
    public class MalformedFrame(public val reason: String) : PieException("Malformed frame: $reason")
}

/** Why the server refused a handle, for a client to branch on. */
public enum class HandleErrorCode(internal val wire: String) {
    /** No such name in this session. */
    UNKNOWN("unknown"),

    /** It existed and is gone; the reason says why. */
    DEAD("dead"),

    /** A session or worker limit: closing a handle makes room. */
    QUOTA("quota"),

    /** The name is taken, or the handle is in a process. */
    IN_USE("in_use"),

    /** The request could never work, whatever the session held. */
    INVALID("invalid"),
    ;

    internal companion object {
        fun named(code: String?): HandleErrorCode? = entries.find { it.wire == code }

        /** The code a refusal's text starts with, if it is one. */
        fun of(reason: String): HandleErrorCode? = named(reason.substringBefore(':'))
    }
}
