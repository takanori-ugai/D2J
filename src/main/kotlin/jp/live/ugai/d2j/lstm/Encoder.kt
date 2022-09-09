package jp.live.ugai.d2j.lstm

import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock

/** The base encoder interface for the encoder-decoder architecture.  */
abstract class Encoder : AbstractBlock() {
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }
}
