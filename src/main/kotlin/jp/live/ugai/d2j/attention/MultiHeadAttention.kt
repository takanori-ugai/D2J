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

fun main() {
    val manager = NDManager.newBaseManager()
//    val numHiddens = 100
    val numHiddens = 5
    val numHeads = 5
    val attention = MultiHeadAttention(numHiddens, numHeads, 0.5f, false)
    val batchSize = 2
    val numQueries = 4
    val numKvpairs = 6
    val validLens = manager.create(floatArrayOf(3.0f, 2.0f))
    val X = manager.ones(Shape(batchSize.toLong(), numQueries.toLong(), numHiddens.toLong()))
    val Y = manager.ones(Shape(batchSize.toLong(), numKvpairs.toLong(), numHiddens.toLong()))

    val ps = ParameterStore(manager, false)
    val input = NDList(X, Y, Y, validLens)
    attention.initialize(manager, DataType.FLOAT32, *input.shapes)
    val result = attention.forward(ps, input, false)
    println(result[0])
}

class MultiHeadAttention(numHiddens: Int, private val numHeads: Int, dropout: Float, useBias: Boolean) :
    AbstractBlock() {
    var attention: DotProductAttention
    private val W_k: Linear
    private val W_q: Linear
    private val W_v: Linear
    private val W_o: Linear

    init {
        attention = DotProductAttention(dropout)
        W_q = Linear.builder().setUnits(numHiddens.toLong()).optBias(useBias).build()
        addChildBlock("W_q", W_q)
        W_k = Linear.builder().setUnits(numHiddens.toLong()).optBias(useBias).build()
        addChildBlock("W_k", W_k)
        W_v = Linear.builder().setUnits(numHiddens.toLong()).optBias(useBias).build()
        addChildBlock("W_v", W_v)
        W_o = Linear.builder().setUnits(numHiddens.toLong()).optBias(useBias).build()
        addChildBlock("W_o", W_o)
        val dropout1 = Dropout.builder().optRate(dropout).build()
        addChildBlock("dropout", dropout1)
    }

    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        // Shape of `queries`, `keys`, or `values`:
        // (`batchSize`, no. of queries or key-value pairs, `numHiddens`)
        // Shape of `validLens`:
        // (`batchSize`,) or (`batchSize`, no. of queries)
        // After transposing, shape of output `queries`, `keys`, or `values`:
        // (`batchSize` * `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        var queries = inputs[0]
        var keys = inputs[1]
        var values = inputs[2]
        var validLens = inputs[3]
        // On axis 0, copy the first item (scalar or vector) for
        // `numHeads` times, then copy the next item, and so on
        if (validLens != null) {
            validLens = validLens.repeat(0, numHeads.toLong())
        }
        queries = Chap10Utils.transposeQkv(W_q.forward(ps, NDList(queries), training, params)[0], numHeads)
        keys = Chap10Utils.transposeQkv(W_k.forward(ps, NDList(keys), training, params)[0], numHeads)
        values = Chap10Utils.transposeQkv(W_v.forward(ps, NDList(values), training, params)[0], numHeads)

        // Shape of `output`: (`batchSize` * `numHeads`, no. of queries,
        // `numHiddens` / `numHeads`)
        val output: NDArray =
            attention
                .forward(ps, NDList(queries, keys, values, validLens), training, params)
                .get(0)

        // Shape of `outputConcat`:
        // (`batchSize`, no. of queries, `numHiddens`)
        val outputConcat: NDArray = Chap10Utils.transposeOutput(output, numHeads)
        return NDList(W_o.forward(ps, NDList(outputConcat), training, params)[0])
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        return arrayOf(inputShapes[0])
    }

    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        manager.newSubManager().use { sub ->
            var queries = sub.zeros(inputShapes[0], dataType)
            var keys = sub.zeros(inputShapes[1], dataType)
            var values = sub.zeros(inputShapes[2], dataType)
            var validLens = sub.zeros(inputShapes[3], dataType)
            validLens = validLens.repeat(0, numHeads.toLong())
            val ps = ParameterStore(sub, false)
            W_q.initialize(manager, dataType, queries.shape)
            W_k.initialize(manager, dataType, keys.shape)
            W_v.initialize(manager, dataType, values.shape)
            queries = Chap10Utils.transposeQkv(W_q.forward(ps, NDList(queries), false)[0], numHeads)
            keys = Chap10Utils.transposeQkv(W_k.forward(ps, NDList(keys), false)[0], numHeads)
            values = Chap10Utils.transposeQkv(W_v.forward(ps, NDList(values), false)[0], numHeads)
            val list = NDList(queries, keys, values, validLens)
            attention.initialize(sub, dataType, *list.shapes)
            val output: NDArray = attention.forward(ps, list, false).head()
            val outputConcat: NDArray = Chap10Utils.transposeOutput(output, numHeads)
            W_o.initialize(manager, dataType, outputConcat.shape)
        }
    }
}
