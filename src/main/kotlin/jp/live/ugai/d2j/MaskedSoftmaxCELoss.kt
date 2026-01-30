package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
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
        val weights =
            labels
                .head()
                .onesLike()
                .expandDims(-1)
                .sequenceMask(labels[1])
        // Remove the states from the labels NDList because otherwise, it will throw an error as SoftmaxCrossEntropyLoss
        // expects only one NDArray for label and one NDArray for prediction
        labels.removeAt(1)
        return super.evaluate(labels, predictions).mul(weights).mean(intArrayOf(1))
    }
}
