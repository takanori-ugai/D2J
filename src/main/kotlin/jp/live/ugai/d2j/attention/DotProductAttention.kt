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

/* Additive attention. */
class AdditiveAttention(numHiddens: Int, dropout: Float) : AbstractBlock() {
    private val W_k: Linear
    private val W_q: Linear
    private val W_v: Linear
    private val dropout: Dropout
    var attentionWeights: NDArray? = null

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

    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        // Shape of the output `queries` and `attentionWeights`:
        // (no. of queries, no. of key-value pairs)
        var queries = inputs[0]
        var keys = inputs[1]
        val values = inputs[2]
        val validLens = inputs[3]
        queries = W_q.forward(ps, NDList(queries), training, params).head()
        keys = W_k.forward(ps, NDList(keys), training, params).head()
        // After dimension expansion, shape of `queries`: (`batchSize`, no. of
        // queries, 1, `numHiddens`) and shape of `keys`: (`batchSize`, 1,
        // no. of key-value pairs, `numHiddens`). Sum them up with
        // broadcasting
        var features = queries.expandDims(2).add(keys.expandDims(1))
        features = features.tanh()
        // There is only one output of `this.W_v`, so we remove the last
        // one-dimensional entry from the shape. Shape of `scores`:
        // (`batchSize`, no. of queries, no. of key-value pairs)
        val result = W_v.forward(ps, NDList(features), training, params).head()
        val scores = result.squeeze(-1)
        attentionWeights = maskedSoftmax(scores, validLens)
        // Shape of `values`: (`batchSize`, no. of key-value pairs, value dimension)
        val list = this.dropout.forward(ps, NDList(attentionWeights), training, params)
        return NDList(list.head().batchDot(values))
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }

    public override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        W_q.initialize(manager, dataType, inputShapes[0])
        W_k.initialize(manager, dataType, inputShapes[1])
        val q = W_q.getOutputShapes(arrayOf(inputShapes[0]))[0].shape
        val k = W_k.getOutputShapes(arrayOf(inputShapes[1]))[0].shape
        val w = Math.max(q[q.size - 2], k[k.size - 2])
        val h = Math.max(q[q.size - 1], k[k.size - 1])
        val shape = longArrayOf(2, 1, w, h)
        W_v.initialize(manager, dataType, Shape(*shape))
        val dropoutShape = LongArray(shape.size - 1)
        System.arraycopy(shape, 0, dropoutShape, 0, dropoutShape.size)
        dropout.initialize(manager, dataType, Shape(*dropoutShape))
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
        val scores = queries.batchDot(keys.swapAxes(1, 2)).div(Math.sqrt(2.0))
        this.attentionWeights = maskedSoftmax(scores, validLens)
        val result = this.dropout0.forward(ps, NDList(this.attentionWeights), false, params)
        return NDList(result[0].batchDot(values))
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        return outputShapes
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        manager.newSubManager().use { sub ->
            val queries = sub.zeros(inputShapes[0], dataType)
            val keys = sub.zeros(inputShapes[1], dataType)
            val scores = queries.batchDot(keys.swapAxes(1, 2))
            val shapes: Array<Shape> = arrayOf(scores.shape)
            dropout0.initialize(manager, dataType, *shapes)
            outputShapes = dropout0.getOutputShapes(shapes)
        }
    }
}
