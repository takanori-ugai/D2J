package jp.live.ugai.d2j.timemachine

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.DataType

/**
 * This class represents an RNN Model implemented from scratch.
 *
 * @property vocabSize The size of the vocabulary.
 * @property numHiddens The number of hidden units.
 * @property params The parameters of the model.
 * @property initState The function to initialize the state.
 * @property forwardFn The forward function of the model.
 */
class RNNModelScratch(
    /** Size of the vocabulary used for one-hot encoding. */
    var vocabSize: Int,
    /** Number of hidden units in the RNN layer. */
    var numHiddens: Int,
    device: Device,
    getParams: (Int, Int, Device) -> NDList,
    initRNNState: (Int, Int, Device) -> NDList,
    forwardFn: (NDArray, NDList, NDList) -> Pair<NDArray, NDList>,
) {
    /**
     * The params.
     */
    var params: NDList = getParams(vocabSize, numHiddens, device)

    /**
     * The initState.
     */
    var initState: (Int, Int, Device) -> NDList = initRNNState

    /**
     * The forwardFn.
     */
    var forwardFn: (NDArray, NDList, NDList) -> Pair<NDArray, NDList> = forwardFn

    /**
     * Performs the forward pass of the model.
     *
     * @param tokens The input data.
     * @param state The state.
     * @return A pair containing the output and the new state.
     */
    fun forward(
        tokens: NDArray,
        state: NDList,
    ): Pair<NDArray, NDList> {
        val input =
            when (tokens.shape.dimension()) {
                2 -> tokens.transpose().toType(DataType.INT64, false).oneHot(vocabSize)
                3 -> {
                    require(tokens.shape.get(2) == vocabSize.toLong()) {
                        "Expected one-hot input with last dimension $vocabSize, got shape ${tokens.shape}."
                    }
                    tokens.toType(DataType.FLOAT32, false)
                }
                else -> {
                    throw IllegalArgumentException(
                        "Expected 2D token indices or 3D one-hot input, got shape ${tokens.shape}.",
                    )
                }
            }
        return forwardFn(input, state, params)
    }

    /**
     * Begins the state of the model.
     *
     * @param batchSize The batch size.
     * @param device The device.
     * @return The initial state.
     */
    fun beginState(
        batchSize: Int,
        device: Device,
    ): NDList = initState(batchSize, numHiddens, device)
}
