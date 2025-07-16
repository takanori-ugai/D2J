package jp.live.ugai.d2j.timemachine

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList

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
    var vocabSize: Int,
    var numHiddens: Int,
    device: Device,
    getParams: (Int, Int, Device) -> NDList,
    initRNNState: (Int, Int, Device) -> NDList,
    forwardFn: (NDArray, NDList, NDList) -> Pair<NDArray, NDList>,
) {
    var params: NDList = getParams(vocabSize, numHiddens, device)
    var initState: (Int, Int, Device) -> NDList = initRNNState
    var forwardFn: (NDArray, NDList, NDList) -> Pair<NDArray, NDList> = forwardFn

    /**
     * Performs the forward pass of the model.
     *
     * @param X The input data.
     * @param state The state.
     * @return A pair containing the output and the new state.
     */
    fun forward(
        X: NDArray,
        state: NDList,
    ): Pair<NDArray, NDList> = forwardFn(X.transpose().oneHot(vocabSize), state, params)

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
