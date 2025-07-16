package jp.live.ugai.d2j.lstm

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.training.ParameterStore
import ai.djl.util.PairList

/**
 * This class represents the base Encoder-Decoder architecture.
 *
 * @property encoder The encoder part of the architecture.
 * @property decoder The decoder part of the architecture.
 */
class EncoderDecoder(
    var encoder: Encoder,
    var decoder: Decoder,
) : AbstractBlock() {
    /**
     * Initializes the Encoder-Decoder architecture.
     */
    init {
        addChildBlock("encoder", encoder)
        addChildBlock("decoder", decoder)
    }

    /**
     * Initializes the child blocks of the Encoder-Decoder architecture.
     *
     * @param manager The NDManager.
     * @param dataType The data type.
     * @param inputShapes The input shapes.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {}

    /**
     * Performs the forward pass of the Encoder-Decoder architecture.
     *
     * @param parameterStore The parameter store.
     * @param inputs The input data.
     * @param training A boolean value indicating whether training is being performed.
     * @param params The parameters.
     * @return The result of the forward pass.
     */
    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val encX = if (inputs.size > 2) NDList(inputs[0], inputs[2]) else NDList(inputs[0])
        val decState = decoder.initState(encoder.forward(parameterStore, encX, training, params))
        return decoder.forward(parameterStore, NDList(inputs[1]).addAll(decState), training, params)
    }

    /**
     * Gets the output shapes of the Encoder-Decoder architecture.
     *
     * @param inputShapes The input shapes.
     * @return The output shapes.
     * @throws UnsupportedOperationException If the method is not implemented.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> = throw UnsupportedOperationException("Not implemented")
}
