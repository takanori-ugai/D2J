package jp.live.ugai.d2j.util

import ai.djl.Device
import ai.djl.engine.Engine
import java.util.function.Function

object Functions {
    // Applies the function `func` to `x` element-wise
    // Returns a new float array with the result
    fun callFunc(
        x: FloatArray,
        func: Function<Float?, Float>,
    ): FloatArray {
        val y = FloatArray(x.size)
        for (i in x.indices) {
            y[i] = func.apply(x[i])
        }
        return y
    }

    // ScatterTrace.builder() does not support float[],
    // so we must convert to a double array first
    fun floatToDoubleArray(x: FloatArray): DoubleArray {
        val ret = DoubleArray(x.size)
        for (i in x.indices) {
            ret[i] = x[i].toDouble()
        }
        return ret
    }

    /**
     * Return the i'th GPU if it exists, otherwise return the CPU
     */
    fun tryGpu(i: Int): Device = if (Engine.getInstance().gpuCount > i) Device.gpu(i) else Device.cpu()

    /**
     * Helper function to later be able to use lambda. Accepts three types for parameters and one
     * for output.
     */
    fun interface TriFunction<T, U, V, W> {
        fun apply(
            t: T,
            u: U,
            v: V,
        ): W
    }

    /**
     * Helper function to later be able to use lambda. Accepts 4 types for parameters and one for
     * output.
     */
    fun interface QuadFunction<T, U, V, W, R> {
        fun apply(
            t: T,
            u: U,
            v: V,
            w: W,
        ): R
    }

    /**
     * Helper function to later be able to use lambda. Doesn't have any type for parameters and has
     * one type for output.
     */
    fun interface SimpleFunction<T> {
        fun apply(): T
    }

    /**
     * Helper function to later be able to use lambda. Accepts one types for parameters and uses
     * void for return.
     */
    fun interface VoidFunction<T> {
        fun apply(t: T)
    }

    /**
     * Helper function to later be able to use lambda. Accepts two types for parameters and uses
     * void for return.
     */
    fun interface VoidTwoFunction<T, U> {
        fun apply(
            t: T,
            u: U,
        )
    }
}
