package jp.live.ugai.d2j.lstm

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock

/**
 * This abstract class represents a Decoder.
 *
 * @property attentionWeights The attention weights.
 */
abstract class Decoder : AbstractBlock() {
    /**
     * The attentionWeights.
     */
    open var attentionWeights: NDArray? = null

    /**
     * Initializes the state of the Decoder.
     *
     * @param encOutputs The encoded outputs.
     * @return The initialized state.
     */
    abstract fun initState(encOutputs: NDList): NDList

    /**
     * Gets the output shapes of the Decoder.
     *
     * @param inputShapes The input shapes.
     * @return The output shapes.
     * @throws UnsupportedOperationException If the method is not implemented.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> =
        throw UnsupportedOperationException("Not implemented")
}
