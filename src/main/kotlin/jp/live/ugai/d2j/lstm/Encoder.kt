package jp.live.ugai.d2j.lstm

import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock

/**
 * This abstract class represents the base Encoder interface for the encoder-decoder architecture.
 */
abstract class Encoder : AbstractBlock() {
    /**
     * Gets the output shapes of the Encoder.
     *
     * @param inputShapes The input shapes.
     * @return The output shapes.
     * @throws UnsupportedOperationException If the method is not implemented.
     */
    abstract override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape>
}
