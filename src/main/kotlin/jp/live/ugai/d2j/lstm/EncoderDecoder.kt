package jp.live.ugai.d2j.lstm

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.training.ParameterStore
import ai.djl.util.PairList

/** The base class for the encoder-decoder architecture.  */
class EncoderDecoder(var encoder: Encoder, var decoder: Decoder) : AbstractBlock() {

    init {
        addChildBlock("encoder", encoder)
        addChildBlock("decoder", decoder)
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {}

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var encX = NDList(inputs[0])
        val decX = NDList(inputs[1])
        if (inputs.size > 2) {
            encX.add(inputs[2])
        }
        val encOutputs = encoder.forward(parameterStore, encX, training, params)
        val decState = decoder.initState(encOutputs)
        val inp = NDList(decX).addAll(decState)
        return decoder.forward(parameterStore, inp, training, params)
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }
}
