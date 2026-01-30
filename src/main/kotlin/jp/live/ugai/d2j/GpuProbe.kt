package jp.live.ugai.d2j

import java.io.BufferedReader
import java.io.InputStreamReader

internal fun logGpu(tag: String) {
    val enabled =
        System.getProperty("D2J_MEMTRACE") == "1" ||
            System.getenv("D2J_MEMTRACE") == "1"
    if (!enabled) return
    try {
        val process =
            ProcessBuilder(
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ).start()
        BufferedReader(InputStreamReader(process.inputStream)).use { reader ->
            val line = reader.readLine()?.trim()
            if (!line.isNullOrEmpty()) {
                println("GPU_MEM $tag : $line MiB")
            }
        }
    } catch (ex: Exception) {
        println("GPU_MEM $tag : unavailable (${ex.javaClass.simpleName})")
    }
}
