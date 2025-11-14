package jp.live.ugai.d2j.util

import ai.djl.Device
import ai.djl.engine.Engine
import java.util.function.Function

/**
 * Utility object providing functional helpers and device selection utilities.
 */
object Functions {

    /**
     * Applies the given function [func] to each element of [x] and returns a new FloatArray.
     */
    fun callFunc(
        x: FloatArray,
        func: Function<Float?, Float>,
    ): FloatArray = FloatArray(x.size) { i -> func.apply(x[i]) }

    /**
     * Converts a FloatArray to a DoubleArray.
     * Useful when APIs require double precision arrays.
     */
    fun floatToDoubleArray(x: FloatArray): DoubleArray = DoubleArray(x.size) { i -> x[i].toDouble() }

    /**
     * Returns the i'th GPU [Device] if available, otherwise returns the CPU [Device].
     */
    fun tryGpu(i: Int): Device =
        if (Engine.getInstance().gpuCount > i) Device.gpu(i) else Device.cpu()

    // --- Functional interfaces for lambda support ---

    /**
     * Functional interface for a function with three parameters and a return value.
     */
    fun interface TriFunction<T, U, V, W> {
        fun apply(t: T, u: U, v: V): W
    }

    /**
     * Functional interface for a function with four parameters and a return value.
     */
    fun interface QuadFunction<T, U, V, W, R> {
        fun apply(t: T, u: U, v: V, w: W): R
    }

    /**
     * Functional interface for a function with no parameters and a return value.
     */
    fun interface SimpleFunction<T> {
        fun apply(): T
    }

    /**
     * Functional interface for a function with one parameter and no return value.
     */
    fun interface VoidFunction<T> {
        fun apply(t: T)
    }

    /**
     * Functional interface for a function with two parameters and no return value.
     */
    fun interface VoidTwoFunction<T, U> {
        fun apply(t: T, u: U)
    }
}
