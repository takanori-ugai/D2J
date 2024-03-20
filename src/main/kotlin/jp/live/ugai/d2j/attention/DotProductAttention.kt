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

fun main() {
    val manager = NDManager.newBaseManager()

    println(
        maskedSoftmax(
            manager.randomUniform(0f, 1f, Shape(2, 2, 4)),
            manager.create(floatArrayOf(2f, 3f))
        )
    )

    println(
        maskedSoftmax(
            manager.randomUniform(0f, 1f, Shape(2, 2, 4)),
            manager.create(arrayOf(floatArrayOf(1f, 3f), floatArrayOf(2f, 4f)))
        )
    )

    var queries = manager.randomNormal(0f, 1f, Shape(2, 1, 20), DataType.FLOAT32)
    val keys = manager.ones(Shape(2, 10, 2))
// The two value matrices in the `values` minibatch are identical
// The two value matrices in the `values` minibatch are identical
    val values = manager.arange(40f).reshape(1, 10, 4).repeat(0, 2)
    val validLens = manager.create(floatArrayOf(2f, 6f))

    val attention = AdditiveAttention(8, 0.1f)
    var input = NDList(queries, keys, values, validLens)
    val ps = ParameterStore(manager, false)
    attention.initialize(manager, DataType.FLOAT32, *input.shapes)
    println(attention.forward(ps, input, false).head())
    println(attention.attentionWeights)

    queries = manager.randomNormal(0f, 1f, Shape(2, 1, 2), DataType.FLOAT32)
    val productAttention = DotProductAttention(0.5f)
    input = NDList(queries, keys, values, validLens)
    productAttention.initialize(manager, DataType.FLOAT32, *input.shapes)
    println(productAttention.forward(ps, input, false).head())
}

/**
 * This class represents the Additive Attention mechanism.
 *
 * @property numHiddens The number of hidden units.
 * @property dropout The dropout rate.
 */
class AdditiveAttention(numHiddens: Int, dropout: Float) : AbstractBlock() {
    private val W_k: Linear
    private val W_q: Linear
    private val W_v: Linear
    private val dropout: Dropout
    var attentionWeights: NDArray? = null

    /**
     * Initializes the Additive Attention mechanism.
     */
    init {
        W_k = Linear.builder().setUnits(numHiddens.toLong()).optBias(false).build()
        addChildBlock("W_k", W_k)
        W_q = Linear.builder().setUnits(numHiddens.toLong()).optBias(false).build()
        addChildBlock("W_q", W_q)
        W_v = Linear.builder().setUnits(1).optBias(false).build()
        addChildBlock("W_v", W_v)
        this.dropout = Dropout.builder().optRate(dropout).build()
        addChildBlock("dropout", this.dropout)
    }

    /**
     * Performs the forward pass of the Additive Attention mechanism.
     *
     * @param ps The parameter store.
     * @param inputs The input data.
     * @param training A boolean value indicating whether training is being performed.
     * @param params The parameters.
     * @return The result of the forward pass.
     */
    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        // Shape of the output `queries` and `attentionWeights`:
        // (no. of queries, no. of key-value pairs)
        val queries = W_q.forward(ps, NDList(inputs[0]), training, params).head()
        val keys = W_k.forward(ps, NDList(inputs[1]), training, params).head()
        val values = inputs[2]
        val validLens = inputs[3]
        // After dimension expansion, shape of `queries`: (`batchSize`, no. of
        // queries, 1, `numHiddens`) and shape of `keys`: (`batchSize`, 1,
        // no. of key-value pairs, `numHiddens`). Sum them up with
        // broadcasting
        val features = queries.expandDims(2).add(keys.expandDims(1)).tanh()
        // There is only one output of `this.W_v`, so we remove the last
        // one-dimensional entry from the shape. Shape of `scores`:
        // (`batchSize`, no. of queries, no. of key-value pairs)
        val scores = W_v.forward(ps, NDList(features), training, params).head().squeeze(-1)

        attentionWeights = maskedSoftmax(scores, validLens)
        // Shape of `values`: (`batchSize`, no. of key-value pairs, value dimension)
        return NDList(this.dropout.forward(ps, NDList(attentionWeights), training, params).head().batchDot(values))
    }

    /**
     * Gets the output shapes of the Additive Attention mechanism.
     *
     * @param inputShapes The input shapes.
     * @return The output shapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        return arrayOf(Shape(outputArrays))
    }

    private var outputArrays: List<Long> = listOf()

    /**
     * Initializes the child blocks of the Additive Attention mechanism.
     *
     * @param manager The NDManager.
     * @param dataType The data type.
     * @param inputShapes The input shapes.
     */
    public override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        W_q.initialize(manager, dataType, inputShapes[0])
        W_k.initialize(manager, dataType, inputShapes[1])
        val outputShapes = arrayOf(
            W_q.getOutputShapes(arrayOf(inputShapes[0]))[0].shape,
            W_k.getOutputShapes(arrayOf(inputShapes[1]))[0].shape
        )
        val w = outputShapes.maxOf { it[it.size - 2] }
        val h = outputShapes.maxOf { it[it.size - 1] }
        val shape = longArrayOf(2, 1, w, h)
        W_v.initialize(manager, dataType, Shape(*shape))
        val dropoutShape = shape.copyOf(shape.size - 1)
        dropout.initialize(manager, dataType, Shape(*dropoutShape))
        outputArrays = listOf(2L, 1L, inputShapes[2][2])
    }
}

/* Scaled dot product attention. */
class DotProductAttention(dropout: Float) : AbstractBlock() {
    private val dropout0: Dropout
    var attentionWeights: NDArray? = null
    private var outputShapes: Array<Shape> = arrayOf<Shape>()

    init {
        this.dropout0 = Dropout.builder().optRate(dropout).build()
        addChildBlock("dropout", this.dropout0)
    }

    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        // Shape of `queries`: (`batchSize`, no. of queries, `d`)
        // Shape of `keys`: (`batchSize`, no. of key-value pairs, `d`)
        // Shape of `values`: (`batchSize`, no. of key-value pairs, value
        // dimension)
        // Shape of `valid_lens`: (`batchSize`,) or (`batchSize`, no. of queries)
        val queries = inputs[0]
        val keys = inputs[1]
        val values = inputs[2]
        val validLens = inputs[3]

        // Swap the last two dimensions of `keys` and perform batchDot
        this.attentionWeights = maskedSoftmax(queries.batchDot(keys.swapAxes(1, 2)).div(Math.sqrt(2.0)), validLens)
        return NDList(this.dropout0.forward(ps, NDList(this.attentionWeights), training, params)[0].batchDot(values))
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        return outputShapes
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        manager.newSubManager().use { sub ->
            val scoresShape = sub.zeros(inputShapes[0], dataType)
                .batchDot(sub.zeros(inputShapes[1], dataType).swapAxes(1, 2)).shape
            dropout0.initialize(manager, dataType, scoresShape)
            outputShapes = dropout0.getOutputShapes(arrayOf(scoresShape))
        }
    }
}
