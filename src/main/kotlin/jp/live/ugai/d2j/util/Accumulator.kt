package jp.live.ugai.d2j.util

/**
 * Accumulates values element-wise in a fixed-size float array.
 *
 * @property data The backing array for accumulation.
 */
class Accumulator(
    n: Int,
) {
    val data = FloatArray(n)

    /**
     * Adds each value from [args] to the corresponding element in [data].
     * Throws [IllegalArgumentException] if lengths do not match.
     */
    fun add(args: FloatArray) {
        require(args.size == data.size) { "Input array size must match accumulator size." }
        for (i in args.indices) {
            data[i] += args[i]
        }
    }

    /** Resets all elements in the accumulator to zero. */
    fun reset() {
        data.fill(0f)
    }

    /** Returns the value at the specified [index]. */
    fun get(index: Int): Float = data[index]
}
