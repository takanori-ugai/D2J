package jp.live.ugai.d2j.lstm

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock

/** The base decoder interface for the encoder-decoder architecture.  */
abstract class Decoder : AbstractBlock() {
    protected var attentionWeights: NDArray? = null
    abstract fun initState(encOutputs: NDList): NDList
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }
}
