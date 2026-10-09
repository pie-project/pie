package org.pieproject.server

internal object NativeCore {
    init {
        System.loadLibrary("pie_jni")
    }

    external fun start(artifact: String, config: String, home: String, listen: String?): Long
    external fun summary(handle: Long): String
    external fun install(handle: Long, program: ByteArray, file: String, version: String?): String
    external fun installLanguage(handle: Long, language: String, component: ByteArray)
    external fun openSession(handle: Long): Int
    external fun closeSession(handle: Long, session: Int)
    external fun sendFrame(handle: Long, session: Int, frame: ByteArray)
    external fun recvFrames(handle: Long, session: Int, maxWaitMs: Int, max: Int): Array<ByteArray>
    external fun shutdown(handle: Long)
}
