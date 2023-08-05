package jp.live.ugai.d2j.attention

import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import jp.live.ugai.d2j.lstm.Decoder

/**
 * An abstract class representing an attention decoder.
 *
 * This class extends the Decoder class and provides a structure for attention-based decoders.
 */
abstract class AttentionDecoder : Decoder() {
    /**
     * A list of pairs, each containing a FloatArray and a Shape, representing the attention weights.
     */
    var attentionWeightArr: MutableList<Pair<FloatArray, Shape>> = mutableListOf()

    /**
     * Initializes the state of the encoder outputs.
     *
     * @param encOutputs The encoder outputs.
     * @return The initialized state.
     */
    abstract override fun initState(encOutputs: NDList): NDList

    /**
     * Gets the output shapes given the input shapes.
     *
     * @param inputShapes The input shapes.
     * @return The output shapes.
     * @throws UnsupportedOperationException If the method is not implemented.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }
}
