package jp.live.ugai.d2j.util

// Saved in the d2l-java package for later use
class StopWatch {
    // Record multiple running times.
    val times: MutableList<Double> = mutableListOf()
    private var tik: Long = 0

    init {
        start()
    }

    fun start() {
        tik = System.nanoTime()
    }

    fun stop(): Double {
        times.add(nanoToSec(System.nanoTime() - tik))
        return times.last()
    }

    // Return average time
    fun avg(): Double {
        return sum() / times.size
    }

    // Return the sum of time
    fun sum(): Double {
        var sum = 0.0
        for (d in times) {
            sum += d
        }
        return sum
    }

    // Return the accumulated times
    fun cumsum(): List<Double> {
        val cumsumList = mutableListOf<Double>()
        var currentSum = 0.0
        for (d in times) {
            currentSum += d
            cumsumList.add(currentSum)
        }
        return cumsumList
    }

    // Convert nano seconds to seconds
    private fun nanoToSec(nanosec: Long): Double {
        return nanosec.toDouble() / 1E9
    }
}
