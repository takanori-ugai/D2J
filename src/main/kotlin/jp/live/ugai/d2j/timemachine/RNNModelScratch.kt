package jp.live.ugai.d2j.timemachine

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList

/** An RNN Model implemented from scratch.  */
class RNNModelScratch(
    var vocabSize: Int,
    var numHiddens: Int,
    device: Device,
    getParams: (Int, Int, Device) -> NDList,
    initRNNState: (Int, Int, Device) -> NDList,
    forwardFn: (NDArray, NDList, NDList) -> Pair<NDArray, NDList>
) {
    var params: NDList = getParams(vocabSize, numHiddens, device)
    var initState: (Int, Int, Device) -> NDList = initRNNState
    var forwardFn: (NDArray, NDList, NDList) -> Pair<NDArray, NDList> = forwardFn

    fun forward(X: NDArray, state: NDList): Pair<NDArray, NDList> {
        var X = X
        X = X.transpose().oneHot(vocabSize)
        return forwardFn(X, state, params)
    }

    fun beginState(batchSize: Int, device: Device): NDList {
        return initState(batchSize, numHiddens, device)
    }
}
