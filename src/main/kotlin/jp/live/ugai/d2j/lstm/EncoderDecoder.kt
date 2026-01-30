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
    /**
     * The encoder.
     */
    var encoder: Encoder,
    /**
     * The decoder.
     */
    var decoder: Decoder,
) : AbstractBlock() {
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
    ) {
        require(inputShapes.size >= 2) {
            "EncoderDecoder expects at least 2 input shapes: encoder input and decoder input."
        }

        val encShapes =
            if (inputShapes.size > 2) {
                arrayOf(inputShapes[0], inputShapes[2])
            } else {
                arrayOf(inputShapes[0])
            }
        when (encShapes.size) {
            1 -> encoder.initialize(manager, dataType, encShapes[0])
            2 -> encoder.initialize(manager, dataType, encShapes[0], encShapes[1])
            else -> throw IllegalArgumentException("Unsupported encoder input shape count: ${encShapes.size}")
        }

        if (inputShapes.size > 2) {
            decoder.initialize(manager, dataType, inputShapes[1], inputShapes[2])
            return
        }

        val encOutShapes =
            try {
                encoder.getOutputShapes(encShapes)
            } catch (ex: UnsupportedOperationException) {
                throw IllegalStateException(
                    "Encoder output shapes are required to initialize the decoder. " +
                        "Provide an explicit state shape or implement encoder output shapes.",
                    ex,
                )
            }

        require(encOutShapes.size >= 2) {
            "Cannot infer decoder state shape from encoder outputs. " +
                "Initialize encoder and decoder separately or provide an explicit state shape."
        }
        decoder.initialize(manager, dataType, inputShapes[1], encOutShapes[1])
    }

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
        require(inputs.size >= 2) {
            "EncoderDecoder forward expects at least 2 inputs: encoder input and decoder input."
        }
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
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        require(inputShapes.size >= 2) {
            "EncoderDecoder expects at least 2 input shapes: encoder input and decoder input."
        }
        if (inputShapes.size > 2) {
            return decoder.getOutputShapes(arrayOf(inputShapes[1], inputShapes[2]))
        }
        val encShapes = arrayOf(inputShapes[0])
        val encOutShapes = encoder.getOutputShapes(encShapes)
        require(encOutShapes.size >= 2) {
            "Cannot infer decoder output shapes from encoder outputs. " +
                "Provide an explicit state shape or implement encoder output shapes."
        }
        return decoder.getOutputShapes(arrayOf(inputShapes[1], encOutShapes[1]))
    }
}
