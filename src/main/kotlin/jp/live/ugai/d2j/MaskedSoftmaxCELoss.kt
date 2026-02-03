package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.DataType
import ai.djl.training.loss.SoftmaxCrossEntropyLoss

/**
 * Computes softmax cross-entropy loss with sequence masking support.
 */
class MaskedSoftmaxCELoss : SoftmaxCrossEntropyLoss() {
    /**
     * Evaluates masked softmax cross-entropy loss for the given labels and predictions.
     */
    override fun evaluate(
        labels: NDList,
        predictions: NDList,
    ): NDArray {
        val yHat = predictions.singletonOrThrow()
        val label =
            if (labels[0].dataType == DataType.INT64) {
                labels[0]
            } else {
                labels[0].toType(DataType.INT64, false)
            }
        val validLen = labels[1]

        val numClasses = yHat.shape[yHat.shape.dimension() - 1].toInt()
        val logProbs = yHat.logSoftmax(-1)
        val oneHot = label.oneHot(numClasses)
        val lossPerToken = logProbs.mul(oneHot).sum(intArrayOf(-1)).neg()
        logProbs.close()
        oneHot.close()

        val weights =
            lossPerToken
                .onesLike()
                .let { ones ->
                    val masked = sequenceMask(ones, validLen)
                    ones.close()
                    masked
                }
        val out = lossPerToken.mul(weights).mean(intArrayOf(1))
        lossPerToken.close()
        weights.close()
        if (label !== labels[0]) {
            label.close()
        }
        return out
    }
}
