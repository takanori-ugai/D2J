package jp.live.ugai.d2j.util

// Saved in the d2l-java package for later use

/**
 * Records elapsed times for repeated operations.
 */
class StopWatch {
    // Record multiple running times.

    /**
     * The times.
     */
    val times: MutableList<Double> = mutableListOf()
    private var tik: Long = 0

    init {
        start()
    }

    /**
     * Starts timing by capturing the current timestamp.
     */
    fun start() {
        tik = System.nanoTime()
    }

    /**
     * Stops timing, records the elapsed duration, and returns it in seconds.
     *
     * @return The elapsed time in seconds for the last interval.
     */
    fun stop(): Double {
        times.add(nanoToSec(System.nanoTime() - tik))
        return times.last()
    }

    // Return average time

    /**
     * Returns the average of all recorded times.
     *
     * @return The arithmetic mean of recorded times in seconds.
     */
    fun avg(): Double = sum() / times.size

    // Return the sum of time

    /**
     * Returns the sum of all recorded times in seconds.
     */
    fun sum(): Double {
        var sum = 0.0
        for (d in times) {
            sum += d
        }
        return sum
    }

    // Return the accumulated times

    /**
     * Returns the cumulative sum of recorded times in seconds.
     */
    fun cumsum(): List<Double> = times.runningFold(0.0) { acc, d -> acc + d }.drop(1)

    // Convert nano seconds to seconds
    private fun nanoToSec(nanosec: Long): Double = nanosec.toDouble() / 1E9
}
