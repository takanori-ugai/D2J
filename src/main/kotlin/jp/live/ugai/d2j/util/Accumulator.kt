package jp.live.ugai.d2j.util

class Accumulator(n: Int) {
    val data = FloatArray(n) { 0f }

    /* Adds a set of numbers to the array */
    fun add(args: FloatArray) {
        for (i in 0 until args.size) {
            data[i] += args[i]
        }
    }

    /* Resets the array */
    fun reset() {
        data.fill(0f)
    }

    /* Returns the data point at the given index */
    fun get(index: Int): Float {
        return data[index]
    }
}
