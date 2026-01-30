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

/**
 * Executes main.
 */
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
    val queries = manager.ones(Shape(batchSize.toLong(), numQueries.toLong(), numHiddens.toLong()))
    val keyValues = manager.ones(Shape(batchSize.toLong(), numKvpairs.toLong(), numHiddens.toLong()))

    val ps = ParameterStore(manager, false)
    val input = NDList(queries, keyValues, keyValues, validLens)
    attention.initialize(
        manager,
        DataType.FLOAT32,
        input.shapes[0],
        input.shapes[1],
        input.shapes[2],
        input.shapes[3],
    )
    val result = attention.forward(ps, input, false)
    println(result[0])
}

/**
 * Represents MultiHeadAttention.
 */
class MultiHeadAttention(
    numHiddens: Int,
    private val numHeads: Int,
    dropout: Float,
    useBias: Boolean,
) : AbstractBlock() {
    /**
     * The attention.
     */
    var attention: DotProductAttention
    private val weightKey: Linear
    private val weightQuery: Linear
    private val weightValue: Linear
    private val weightOutput: Linear
    private val projDropout: Dropout

    init {
        require(numHiddens % numHeads == 0) {
            "numHiddens ($numHiddens) must be divisible by numHeads ($numHeads)"
        }
        attention = DotProductAttention(dropout)
        weightQuery =
            Linear
                .builder()
                .setUnits(numHiddens.toLong())
                .optBias(useBias)
                .build()
        addChildBlock("W_q", weightQuery)
        weightKey =
            Linear
                .builder()
                .setUnits(numHiddens.toLong())
                .optBias(useBias)
                .build()
        addChildBlock("W_k", weightKey)
        weightValue =
            Linear
                .builder()
                .setUnits(numHiddens.toLong())
                .optBias(useBias)
                .build()
        addChildBlock("W_v", weightValue)
        weightOutput =
            Linear
                .builder()
                .setUnits(numHiddens.toLong())
                .optBias(useBias)
                .build()
        addChildBlock("W_o", weightOutput)
        this.projDropout = Dropout.builder().optRate(dropout).build()
        addChildBlock("proj_dropout", this.projDropout)
    }

    /**
     * Executes forwardInternal.
     */
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
        var validLens = if (inputs.size > 3) inputs[3] else null
        // On axis 0, copy the first item (scalar or vector) for
        // `numHeads` times, then copy the next item, and so on
        if (validLens != null) {
            validLens = validLens.repeat(0, numHeads.toLong())
        }
        queries = Chap10Utils.transposeQkv(weightQuery.forward(ps, NDList(queries), training, params)[0], numHeads)
        keys = Chap10Utils.transposeQkv(weightKey.forward(ps, NDList(keys), training, params)[0], numHeads)
        values = Chap10Utils.transposeQkv(weightValue.forward(ps, NDList(values), training, params)[0], numHeads)

        // Shape of `output`: (`batchSize` * `numHeads`, no. of queries,
        // `numHiddens` / `numHeads`)
        val attnInputs =
            if (validLens == null) {
                NDList(queries, keys, values)
            } else {
                NDList(queries, keys, values, validLens)
            }
        val output: NDArray = attention.forward(ps, attnInputs, training, params).get(0)

        // Shape of `outputConcat`:
        // (`batchSize`, no. of queries, `numHiddens`)
        val outputConcat: NDArray = Chap10Utils.transposeOutput(output, numHeads)
        val projected = weightOutput.forward(ps, NDList(outputConcat), training, params)[0]
        return projDropout.forward(ps, NDList(projected), training, params)
    }

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> = arrayOf(inputShapes[0])

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        manager.newSubManager().use { sub ->
            var queries = sub.zeros(inputShapes[0], dataType)
            var keys = sub.zeros(inputShapes[1], dataType)
            var values = sub.zeros(inputShapes[2], dataType)
            val hasValidLens = inputShapes.size > 3
            var validLens = if (hasValidLens) sub.zeros(inputShapes[3], dataType) else null
            if (validLens != null) {
                validLens = validLens.repeat(0, numHeads.toLong())
            }
            val ps = ParameterStore(sub, false)
            weightQuery.initialize(manager, dataType, queries.shape)
            weightKey.initialize(manager, dataType, keys.shape)
            weightValue.initialize(manager, dataType, values.shape)
            queries = Chap10Utils.transposeQkv(weightQuery.forward(ps, NDList(queries), false)[0], numHeads)
            keys = Chap10Utils.transposeQkv(weightKey.forward(ps, NDList(keys), false)[0], numHeads)
            values = Chap10Utils.transposeQkv(weightValue.forward(ps, NDList(values), false)[0], numHeads)
            val list =
                if (validLens == null) {
                    NDList(queries, keys, values)
                } else {
                    NDList(queries, keys, values, validLens)
                }
            if (list.size == 3) {
                attention.initialize(sub, dataType, list[0].shape, list[1].shape, list[2].shape)
            } else {
                attention.initialize(sub, dataType, list[0].shape, list[1].shape, list[2].shape, list[3].shape)
            }
            val output: NDArray = attention.forward(ps, list, false).head()
            val outputConcat: NDArray = Chap10Utils.transposeOutput(output, numHeads)
            weightOutput.initialize(manager, dataType, outputConcat.shape)
        }
    }
}
