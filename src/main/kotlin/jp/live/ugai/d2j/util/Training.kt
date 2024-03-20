package jp.live.ugai.d2j.util

import ai.djl.metric.Metrics
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.training.EasyTrain
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Batch
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker

object Training {

    fun linreg(X: NDArray, w: NDArray, b: NDArray): NDArray {
        return X.dot(w).add(b)
    }

    fun squaredLoss(yHat: NDArray, y: NDArray): NDArray {
        return (yHat.sub(y.reshape(yHat.shape)))
            .mul((yHat.sub(y.reshape(yHat.shape))))
            .div(2)
    }

    fun sgd(params: NDList, lr: Float, batchSize: Int) {
        val lrt = Tracker.fixed(lr)
        val opt = Optimizer.sgd().setLearningRateTracker(lrt).build()
        for (param in params) {
            // Update param in place.
            // param = param - param.gradient * lr / batchSize
            // val ind = params.indexOf(param)
            // params.rep
            // params.set(ind, param.sub(param.getGradient().mul(lr).div(batchSize)))
            opt.update(param.toString(), param, param.gradient.div(batchSize))
//            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }

    /**
     * Allows to do gradient calculations on a subManager. This is very useful when you are training
     * on a lot of epochs. This subManager could later be closed and all NDArrays generated from the
     * calculations in this function will be cleared from memory when subManager is closed. This is
     * always a great practice but the impact is most notable when there is lot of data on various
     * epochs.
     */
    fun sgd(params: NDList, lr: Float, batchSize: Int, subManager: NDManager) {
        sgd(params, lr, batchSize)
    }

    fun accuracy(yHat: NDArray, y: NDArray): Float {
        // Check size of 1st dimension greater than 1
        // to see if we have multiple samples
        if (yHat.shape.size(1) > 1) {
            // Argmax gets index of maximum args for given axis 1
            // Convert yHat to same dataType as y (int32)
            // Sum up number of true entries
            return yHat.argMax(1)
                .toType(DataType.INT32, false)
                .eq(y.toType(DataType.INT32, false))
                .sum()
                .toType(DataType.FLOAT32, false)
                .getFloat()
        }
        return yHat.toType(DataType.INT32, false)
            .eq(y.toType(DataType.INT32, false))
            .sum()
            .toType(DataType.FLOAT32, false)
            .getFloat()
    }

    fun trainingChapter6(
        trainIter: ArrayDataset,
        testIter: ArrayDataset,
        numEpochs: Int,
        trainer: Trainer,
        evaluatorMetrics: MutableMap<String, DoubleArray>
    ): Double {
        trainer.metrics = Metrics()

        EasyTrain.fit(trainer, numEpochs, trainIter, testIter)

        val metrics = trainer.metrics

        trainer.evaluators
            .forEach { evaluator ->
                evaluatorMetrics.put(
                    "train_epoch_" + evaluator.name,
                    metrics.getMetric("train_epoch_" + evaluator.name).map { x -> x.value }.toDoubleArray()
                )
                evaluatorMetrics.put(
                    "validate_epoch_" + evaluator.name,
                    metrics.getMetric("validate_epoch_" + evaluator.name).map { x -> x.value }.toDoubleArray()
                )
            }

        return metrics.mean("epoch")
    }

    /* Softmax-regression-scratch */
    fun evaluateAccuracy(net: (NDArray) -> NDArray, dataIterator: Iterable<Batch>): Float {
        val metric = Accumulator(2) // numCorrectedExamples, numExamples
        for (batch in dataIterator) {
            val X = batch.data.head()
            val y = batch.labels.head()
            metric.add(floatArrayOf(accuracy(net(X), y), y.size().toFloat()))
            batch.close()
        }
        return metric.get(0) / metric.get(1)
    }
    /* End Softmax-regression-scratch */

    /* MLP */
    /* Evaluate the loss of a model on the given dataset */
    fun evaluateLoss(
        net: (NDArray) -> NDArray,
        dataIterator: Iterable<Batch>,
        loss: (NDArray, NDArray) -> NDArray
    ): Float {
        val metric = Accumulator(2) // sumLoss, numExamples

        for (batch in dataIterator) {
            val X = batch.data.head()
            val y = batch.labels.head()
            metric.add(
                floatArrayOf(loss(net(X), y).sum().getFloat(), y.size().toFloat())
            )
            batch.close()
        }
        return metric.get(0) / metric.get(1)
    }
    /* End MLP */
}
