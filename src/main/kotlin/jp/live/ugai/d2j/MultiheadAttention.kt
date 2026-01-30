package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import jp.live.ugai.d2j.attention.MultiHeadAttention

/**
 * Executes main.
 */
fun main() {
    val manager = NDManager.newBaseManager()

    fun transposeQkv(
        input: NDArray,
        numHeads: Int,
    ): NDArray? {
        // Shape of input `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHiddens`).
        // Shape of output `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHeads`,
        // `numHiddens` / `numHeads`)
        var tensor = input
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], numHeads.toLong(), -1)

        // Shape of output `X`:
        // (`batchSize`, `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        tensor = tensor.transpose(0, 2, 1, 3)

        // Shape of `output`:
        // (`batchSize` * `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        return tensor.reshape(-1, tensor.shape[2], tensor.shape[3])
    }

    fun transposeOutput(
        input: NDArray,
        numHeads: Int,
    ): NDArray? {
        var tensor = input
        tensor = tensor.reshape(-1, numHeads.toLong(), tensor.shape[1], tensor.shape[2])
        tensor = tensor.transpose(0, 2, 1, 3)
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
    }

    val numHiddens = 100
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
    println(result[0].shape)
}

/**
 * Placeholder for the multi-head attention example (see the main function above).
 */
internal class MultiheadAttention
