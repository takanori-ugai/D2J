package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.training.loss.SoftmaxCrossEntropyLoss

/**
 * The softmax cross-entropy loss with masks.
 */
class MaskedSoftmaxCELoss : SoftmaxCrossEntropyLoss() {
    /** {@inheritDoc}  */
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
