package jp.live.ugai.d2j.attention

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.types.Shape

object Chap10Utils {
    fun maskedSoftmax(_X: NDArray, _validLens: NDArray): NDArray {
        /* Perform softmax operation by masking elements on the last axis. */
        // `X`: 3D tensor, `validLens`: 1D or 2D tensor
        var X = _X
        var validLens = _validLens ?: return X.softmax(-1)
        val shape: Shape = X.shape
        if (validLens.shape.dimension() == 0) {
            return X.softmax(-1).reshape(shape)
        }
        if (validLens.shape.dimension() == 1) {
            validLens = validLens.repeat(shape.get(1))
        } else {
            validLens = validLens.reshape(-1)
        }
        // On the last axis, replace masked elements with a very large negative
        // value, whose exponentiation outputs 0
        X = X.reshape(Shape(-1, shape.get(shape.dimension() - 1)))
            .sequenceMask(validLens, -1E6.toFloat())
        return X.softmax(-1).reshape(shape)
    }

    fun transposeQkv(_X: NDArray, numHeads: Int): NDArray {
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

    fun transposeOutput(_X: NDArray, numHeads: Int): NDArray {
        var X = _X
        X = X.reshape(-1, numHeads.toLong(), X.shape[1], X.shape[2])
        X = X.transpose(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    }
}
