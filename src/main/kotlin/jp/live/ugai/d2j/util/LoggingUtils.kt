package jp.live.ugai.d2j.util

/**
 * Logging configuration helpers for DJL examples.
 */
object LoggingUtils {
    /**
     * Sets standard DJL logging system properties.
     */
    fun setDjlLoggingProperties() {
        System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
        System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
        System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
        System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
        System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
        System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")
    }
}
