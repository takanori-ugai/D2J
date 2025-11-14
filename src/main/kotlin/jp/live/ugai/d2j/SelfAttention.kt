package jp.live.ugai.d2j

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import jp.live.ugai.d2j.attention.MultiHeadAttention

/**
 * Self-attention block that uses MultiHeadAttention.
 * The query, key, and value are all the same input.
 */
class SelfAttention(
    numHiddens: Int,
    numHeads: Int,
    dropout: Float,
) : AbstractBlock() {
    private val attention: MultiHeadAttention

    init {
        attention = MultiHeadAttention(numHiddens, numHeads, dropout, false)
        addChildBlock("attention", attention)
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val x = inputs[0]
        val validLens = if (inputs.size > 1) inputs[1] else null
        val attentionInputs = NDList(x, x, x)
        if (validLens != null) {
            attentionInputs.add(validLens)
        }
        return attention.forward(parameterStore, attentionInputs, training, params)
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        // The output shape is the same as the input shape.
        return arrayOf(inputShapes[0])
    }

    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        val shape = inputShapes[0]
        if (inputShapes.size > 1) {
            attention.initialize(manager, dataType, shape, shape, shape, inputShapes[1])
        } else {
            attention.initialize(manager, dataType, shape, shape, shape)
        }
    }
}

fun main() {
    val manager = NDManager.newBaseManager()

    // --- Test SelfAttention ---
    val numHiddens = 100
    val numHeads = 5
    val selfAttention = SelfAttention(numHiddens, numHeads, 0.5f)

    val batchSize = 2
    val numSteps = 4 // Also known as number of queries/keys/values
    val validLens = manager.create(floatArrayOf(3.0f, 2.0f))
    val x = manager.ones(Shape(batchSize.toLong(), numSteps.toLong(), numHiddens.toLong()))

    val ps = ParameterStore(manager, false)
    val inputs = NDList(x, validLens)
    selfAttention.initialize(manager, DataType.FLOAT32, *inputs.shapes)

    val result = selfAttention.forward(ps, inputs, false)
    println("SelfAttention output shape: ${result[0].shape}")
    println("The output shape should be the same as the input shape: ${x.shape}")
}
