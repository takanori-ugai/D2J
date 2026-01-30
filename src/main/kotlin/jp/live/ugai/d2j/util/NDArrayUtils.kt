package jp.live.ugai.d2j.util

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

/**
 * Utility helpers for NDArray initialization.
 */
object NDArrayUtils {
    /**
     * Creates a random normal NDArray with mean 0 and stddev 0.01.
     *
     * @param manager The NDManager used to allocate the array.
     * @param shape The output shape.
     * @param device The device to place the array on.
     * @return A newly allocated NDArray with random normal values.
     */
    fun normal(
        manager: NDManager,
        shape: Shape,
        device: Device,
    ): NDArray = manager.randomNormal(0.0f, 0.01f, shape, DataType.FLOAT32, device)

    /**
     * Builds a parameter triplet for recurrent layers: input weights, hidden weights, and bias.
     *
     * @param manager The NDManager used to allocate the arrays.
     * @param numInputs The input feature size.
     * @param numHiddens The hidden state size.
     * @param device The device to place the arrays on.
     * @return NDList containing (W_x, W_h, b) NDArrays.
     */
    fun three(
        manager: NDManager,
        numInputs: Int,
        numHiddens: Int,
        device: Device,
    ): NDList =
        NDList(
            normal(manager, Shape(numInputs.toLong(), numHiddens.toLong()), device),
            normal(manager, Shape(numHiddens.toLong(), numHiddens.toLong()), device),
            manager.zeros(Shape(numHiddens.toLong()), DataType.FLOAT32, device),
        )
}
