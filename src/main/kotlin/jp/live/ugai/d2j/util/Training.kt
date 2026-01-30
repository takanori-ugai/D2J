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

/**
 * Retrieves a system property as a Long value, returning a default if not set or invalid.
 */
fun getLong(
    propertyName: String,
    defaultValue: Long,
): Long = System.getProperty(propertyName)?.toLongOrNull() ?: defaultValue

/**
 * Singleton for Training.
 */
object Training {
    fun linreg(
        features: NDArray,
        w: NDArray,
        b: NDArray,
    ): NDArray = features.dot(w).add(b)

    fun squaredLoss(
        yHat: NDArray,
        y: NDArray,
    ): NDArray =
        (yHat.sub(y.reshape(yHat.shape)))
            .mul((yHat.sub(y.reshape(yHat.shape))))
            .div(2)

    fun sgd(
        params: NDList,
        lr: Float,
        batchSize: Int,
    ) {
        val lrt = Tracker.fixed(lr)
        val opt = Optimizer.sgd().setLearningRateTracker(lrt).build()
        for ((index, param) in params.withIndex()) {
            // Update param in place.
            // param = param - param.gradient * lr / batchSize
            // val ind = params.indexOf(param)
            // params.rep
            // params.set(ind, param.sub(param.getGradient().mul(lr).div(batchSize)))
            opt.update("param_$index", param, param.gradient.div(batchSize))
//            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }

    /**
     * Allows gradient calculations on a subManager. This is useful when training many epochs;
     * arrays created for gradients are attached to the subManager and cleared when it is closed.
     */
    fun sgd(
        params: NDList,
        lr: Float,
        batchSize: Int,
        subManager: NDManager,
    ) {
        val lrt = Tracker.fixed(lr)
        val opt = Optimizer.sgd().setLearningRateTracker(lrt).build()
        for ((index, param) in params.withIndex()) {
            // Update param in place.
            // param = param - param.gradient * lr / batchSize
            val gradient = param.gradient
            gradient.attach(subManager)
            opt.update("param_$index", param, gradient.div(batchSize))
        }
    }

    fun accuracy(
        yHat: NDArray,
        y: NDArray,
    ): Float {
        // If yHat is 2D with class scores, use argMax; otherwise compare directly.
        return if (yHat.shape.dimension() > 1 && yHat.shape.get(1) > 1) {
            yHat
                .argMax(1)
                .toType(DataType.INT32, false)
                .eq(y.toType(DataType.INT32, false))
                .sum()
                .toType(DataType.FLOAT32, false)
                .getFloat()
        } else {
            yHat
                .toType(DataType.INT32, false)
                .eq(y.toType(DataType.INT32, false))
                .sum()
                .toType(DataType.FLOAT32, false)
                .getFloat()
        }
    }

    fun trainingChapter6(
        trainIter: ArrayDataset,
        testIter: ArrayDataset,
        numEpochs: Int,
        trainer: Trainer,
        evaluatorMetrics: MutableMap<String, DoubleArray>,
    ): Double {
        trainer.metrics = Metrics()

        EasyTrain.fit(trainer, numEpochs, trainIter, testIter)

        val metrics = trainer.metrics

        trainer.evaluators
            .forEach { evaluator ->
                val trainMetric = metrics.getMetric("train_epoch_" + evaluator.name)
                if (trainMetric != null) {
                    evaluatorMetrics.put(
                        "train_epoch_" + evaluator.name,
                        trainMetric.map { x -> x.value }.toDoubleArray(),
                    )
                }
                val validateMetric = metrics.getMetric("validate_epoch_" + evaluator.name)
                if (validateMetric != null) {
                    evaluatorMetrics.put(
                        "validate_epoch_" + evaluator.name,
                        validateMetric.map { x -> x.value }.toDoubleArray(),
                    )
                }
            }

        return metrics.mean("epoch")
    }

    // Softmax-regression-scratch
    fun evaluateAccuracy(
        net: (NDArray) -> NDArray,
        dataIterator: Iterable<Batch>,
    ): Float {
        val metric = Accumulator(2) // numCorrectedExamples, numExamples
        for (batch in dataIterator) {
            try {
                val X = batch.data.head()
                val y = batch.labels.head()
                metric.add(floatArrayOf(accuracy(net(X), y), y.size().toFloat()))
            } finally {
                batch.close()
            }
        }
        return metric.get(0) / metric.get(1)
    }
    // End Softmax-regression-scratch

    /**
     * MLP
     * Evaluate the loss of a model on the given dataset
     */
    fun evaluateLoss(
        net: (NDArray) -> NDArray,
        dataIterator: Iterable<Batch>,
        loss: (NDArray, NDArray) -> NDArray,
    ): Float {
        val metric = Accumulator(2) // sumLoss, numExamples

        for (batch in dataIterator) {
            try {
                val X = batch.data.head()
                val y = batch.labels.head()
                metric.add(
                    floatArrayOf(loss(net(X), y).sum().getFloat(), y.size().toFloat()),
                )
            } finally {
                batch.close()
            }
        }
        return metric.get(0) / metric.get(1)
    }
    // End MLP
}
