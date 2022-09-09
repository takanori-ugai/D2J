package jp.live.ugai.d2j.lstm

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.training.ParameterStore
import ai.djl.util.PairList

/** The base class for the encoder-decoder architecture.  */
class EncoderDecoder(protected var encoder: Encoder, decoder: Decoder) : AbstractBlock() {
    protected var decoder: Decoder

    init {
        addChildBlock("encoder", encoder)
        this.decoder = decoder
        addChildBlock("decoder", this.decoder)
    }

    override fun initializeChildBlocks(manager: NDManager?, dataType: DataType, vararg inputShapes: Shape) {}

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>
    ): NDList {
        val encX = inputs[0]
        val decX = inputs[1]
        val encOutputs = encoder.forward(parameterStore, NDList(encX), training, params)
        val decState = decoder.initState(encOutputs)
        return decoder.forward(parameterStore, NDList(decX).addAll(decState), training, params)
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }
}
