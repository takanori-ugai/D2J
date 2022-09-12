package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import jp.live.ugai.d2j.attention.MultiHeadAttention

fun main() {
    val manager = NDManager.newBaseManager()

    fun transposeQkv(_X: NDArray, numHeads: Int): NDArray? {
        // Shape of input `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHiddens`).
        // Shape of output `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHeads`,
        // `numHiddens` / `numHeads`)
        var X = _X
        X = X.reshape(X.shape[0], X.shape[1], numHeads.toLong(), -1)

        // Shape of output `X`:
        // (`batchSize`, `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        X = X.transpose(0, 2, 1, 3)

        // Shape of `output`:
        // (`batchSize` * `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        return X.reshape(-1, X.shape[2], X.shape[3])
    }

    fun transposeOutput(_X: NDArray, numHeads: Int): NDArray? {
        var X = _X
        X = X.reshape(-1, numHeads.toLong(), X.shape[1], X.shape[2])
        X = X.transpose(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    }

    val numHiddens = 100
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
    println(result[0].shape)
}

class MultiheadAttention
