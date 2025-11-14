package jp.live.ugai.d2j.attention

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.Dropout
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import jp.live.ugai.d2j.attention.Chap10Utils.maskedSoftmax

/**
 * Implements the Additive Attention mechanism.
 *
 * @property numHiddens Number of hidden units for projections.
 * @property dropout Dropout rate for attention weights.
 */
class AdditiveAttention(
    numHiddens: Int,
    dropout: Float,
) : AbstractBlock() {
    private val wK =
        Linear
            .builder()
            .setUnits(numHiddens.toLong())
            .optBias(false)
            .build()
    private val wQ =
        Linear
            .builder()
            .setUnits(numHiddens.toLong())
            .optBias(false)
            .build()
    private val wV =
        Linear
            .builder()
            .setUnits(1)
            .optBias(false)
            .build()
    private val dropoutLayer: Dropout = Dropout.builder().optRate(dropout).build()
    var attentionWeights: NDArray? = null

    init {
        addChildBlock("W_k", wK)
        addChildBlock("W_q", wQ)
        addChildBlock("W_v", wV)
        addChildBlock("dropout", dropoutLayer)
    }

    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val queries = wQ.forward(ps, NDList(inputs[0]), training, params).head()
        val keys = wK.forward(ps, NDList(inputs[1]), training, params).head()
        val values = inputs[2]
        val validLens = inputs[3]

        val features = queries.expandDims(2).add(keys.expandDims(1)).tanh()
        val scores = wV.forward(ps, NDList(features), training, params).head().squeeze(-1)

        attentionWeights = maskedSoftmax(scores, validLens)
        val droppedAttention = dropoutLayer.forward(ps, NDList(attentionWeights), training, params).head()
        return NDList(droppedAttention.matMul(values))
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        val queriesShape = inputShapes[0]
        val valuesShape = inputShapes[2]
        return arrayOf(Shape(queriesShape[0], queriesShape[1], valuesShape[2]))
    }

    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        wQ.initialize(manager, dataType, inputShapes[0])
        wK.initialize(manager, dataType, inputShapes[1])

        // Determine the shape for wV based on the outputs of wQ
        val qShape = wQ.getOutputShapes(arrayOf(inputShapes[0]))[0]
        val vFeatureSize = qShape[qShape.dimension() - 1]
        wV.initialize(manager, dataType, Shape(1, vFeatureSize))

        // Initialize dropout with a representative shape
        val scoresShape = Shape(inputShapes[0][0], inputShapes[0][1], inputShapes[1][1])
        dropoutLayer.initialize(manager, dataType, scoresShape)
    }
}

/**
 * Implements scaled dot product attention.
 *
 * @property dropout Dropout rate for attention weights.
 */
class DotProductAttention(
    dropout: Float,
) : AbstractBlock() {
    private val dropoutLayer: Dropout = Dropout.builder().optRate(dropout).build()
    var attentionWeights: NDArray? = null

    init {
        addChildBlock("dropout", dropoutLayer)
    }

    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val queries = inputs[0]
        val keys = inputs[1]
        val values = inputs[2]
        val validLens = inputs[3]

        val d = queries.shape[queries.shape.dimension() - 1].toDouble()
        val scores = queries.matMul(keys.swapAxes(1, 2)).div(Math.sqrt(d))

        attentionWeights = maskedSoftmax(scores, validLens)
        val droppedAttention = dropoutLayer.forward(ps, NDList(attentionWeights), training, params)[0]
        return NDList(droppedAttention.matMul(values))
    }

    override fun getOutputShapes(shapes: Array<Shape>): Array<Shape> {
        val queriesShape = shapes[0]
        val valuesShape = shapes[2]
        return arrayOf(Shape(queriesShape[0], queriesShape[1], valuesShape[2]))
    }

    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        // Calculate the shape of the attention scores for dropout initialization
        val scoresShape = Shape(inputShapes[0][0], inputShapes[0][1], inputShapes[1][1])
        dropoutLayer.initialize(manager, dataType, scoresShape)
    }

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            val manager = NDManager.newBaseManager()

            // Test maskedSoftmax
            println("--- Testing maskedSoftmax ---")
            println(
                maskedSoftmax(
                    manager.randomUniform(0f, 1f, Shape(2, 2, 4)),
                    manager.create(floatArrayOf(2f, 3f)),
                ),
            )
            println(
                maskedSoftmax(
                    manager.randomUniform(0f, 1f, Shape(2, 2, 4)),
                    manager.create(arrayOf(floatArrayOf(1f, 3f), floatArrayOf(2f, 4f))),
                ),
            )

            // Test AdditiveAttention
            println("\n--- Testing AdditiveAttention ---")
            val queriesA = manager.randomNormal(0f, 1f, Shape(2, 1, 20), DataType.FLOAT32)
            val keysA = manager.ones(Shape(2, 10, 2))
            val valuesA = manager.arange(40f).reshape(1, 10, 4).repeat(0, 2)
            val validLensA = manager.create(floatArrayOf(2f, 6f))

            val additiveAttention = AdditiveAttention(8, 0.1f)
            val inputA = NDList(queriesA, keysA, valuesA, validLensA)
            val ps = ParameterStore(manager, false)
            additiveAttention.initialize(manager, DataType.FLOAT32, *inputA.shapes)
            println("Output shape: ${additiveAttention.forward(ps, inputA, false).head().shape}")
            println("Attention weights shape: ${additiveAttention.attentionWeights?.shape}")

            // Test DotProductAttention
            println("\n--- Testing DotProductAttention ---")
            val queriesD = manager.randomNormal(0f, 1f, Shape(2, 1, 2), DataType.FLOAT32)
            val dotProductAttention = DotProductAttention(0.5f)
            val inputD = NDList(queriesD, keysA, valuesA, validLensA)
            dotProductAttention.initialize(manager, DataType.FLOAT32, *inputD.shapes)
            println("Output shape: ${dotProductAttention.forward(ps, inputD, false).head().shape}")
            println("Attention weights shape: ${dotProductAttention.attentionWeights?.shape}")
        }
    }
}
