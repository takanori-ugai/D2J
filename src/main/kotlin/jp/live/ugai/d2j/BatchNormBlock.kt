package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.Blocks
import ai.djl.nn.Parameter
import ai.djl.nn.SequentialBlock
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.core.Linear
import ai.djl.nn.pooling.Pool
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import jp.live.ugai.d2j.util.Training
import jp.live.ugai.d2j.util.getLong

class BatchNormBlock(
    private val numFeatures: Int,
    numDimensions: Int,
) : AbstractBlock() {
    private var movingMean: NDArray
    private var movingVar: NDArray
    private var gamma: Parameter
    private var beta: Parameter

    init {
        val shape =
            if (numDimensions == 2) {
                Shape(1, numFeatures.toLong())
            } else {
                Shape(1, numFeatures.toLong(), 1, 1)
            }
        gamma =
            addParameter(
                Parameter
                    .builder()
                    .setName("gamma")
                    .setType(Parameter.Type.GAMMA)
                    .optShape(shape)
                    .build(),
            )
        beta =
            addParameter(
                Parameter
                    .builder()
                    .setName("beta")
                    .setType(Parameter.Type.BETA)
                    .optShape(shape)
                    .build(),
            )
        val manager = NDManager.newBaseManager()
        movingMean = manager.zeros(shape)
        movingVar = manager.zeros(shape)
    }

    private fun batchNormUpdate(
        X: NDArray,
        gamma: NDArray,
        beta: NDArray,
        movingMean0: NDArray,
        movingVar0: NDArray,
        eps: Float,
        momentum: Float,
        isTraining: Boolean,
    ): NDList {
        var movingMean = movingMean0
        var movingVar = movingVar0
        movingMean.manager.newSubManager().use { subManager ->
            movingMean.attach(subManager)
            movingVar.attach(subManager)
            val xHat: NDArray
            if (!isTraining) {
                xHat = X.sub(movingMean).div(movingVar.add(eps).sqrt())
            } else {
                // Averages over all axes except for the 'features' axis (axis 1)
                val axesToReduce = (0 until X.shape.dimension()).filter { it != 1 }.toIntArray()

                // The DJL PyTorch engine only supports mean over a single axis, so we fold/chain the calls
                val mean = axesToReduce.fold(X) { acc, axis -> acc.mean(intArrayOf(axis), true) }
                val vari = axesToReduce.fold(X.sub(mean).pow(2)) { acc, axis -> acc.mean(intArrayOf(axis), true) }

                xHat = X.sub(mean).div(vari.add(eps).sqrt())
                movingMean = movingMean.mul(momentum).add(mean.mul(1.0f - momentum))
                movingVar = movingVar.mul(momentum).add(vari.mul(1.0f - momentum))
            }
            val Y = xHat.mul(gamma).add(beta)
            movingMean.attach(subManager.parentManager)
            movingVar.attach(subManager.parentManager)
            return NDList(Y, movingMean, movingVar)
        }
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val result =
            batchNormUpdate(
                inputs.singletonOrThrow(),
                gamma.array,
                beta.array,
                movingMean,
                movingVar,
                1e-12f,
                0.9f,
                training,
            )
        if (training) {
            movingMean.close()
            movingVar.close()
        }
        movingMean = result[1]
        movingVar = result[2]
        return NDList(result[0])
    }

    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        // Batch norm does not change the shape of the input
        return inputs
    }

    override fun toString(): String = "BatchNormBlock(numFeatures=$numFeatures)"

    companion object {
        private const val BATCH_SIZE = 256
        private const val NUM_EPOCHS = 10
        private const val LEARNING_RATE = 0.1f

        @JvmStatic
        fun main(args: Array<String>) {
            setSystemProperties()

            val trainIter = prepareDataset(Dataset.Usage.TRAIN, BATCH_SIZE)
            val testIter = prepareDataset(Dataset.Usage.TEST, BATCH_SIZE)

            val model = Model.newInstance("batch-norm-block")
            model.block = prepareModelBlock()

            val loss = Loss.softmaxCrossEntropyLoss()
            val tracker = Tracker.fixed(LEARNING_RATE)
            val optimizer = Optimizer.sgd().setLearningRateTracker(tracker).build()

            val config =
                DefaultTrainingConfig(loss)
                    .optOptimizer(optimizer)
                    .optDevices(Engine.getInstance().getDevices(1))
                    .addEvaluator(Accuracy())
                    .addTrainingListeners(*TrainingListener.Defaults.logging())

            val trainer = model.newTrainer(config)
            trainer.initialize(Shape(1, 1, 28, 28))

            val evaluatorMetrics = mutableMapOf<String, DoubleArray>()
            val avgTrainTimePerEpoch =
                Training.trainingChapter6(trainIter, testIter, NUM_EPOCHS, trainer, evaluatorMetrics)

            val trainLoss = evaluatorMetrics["train_epoch_SoftmaxCrossEntropyLoss"]
            val trainAccuracy = evaluatorMetrics["train_epoch_Accuracy"]
            val testAccuracy = evaluatorMetrics["validate_epoch_Accuracy"]

            println(
                "loss %.3f, train acc %.3f, test acc %.3f".format(
                    trainLoss!![NUM_EPOCHS - 1],
                    trainAccuracy!![NUM_EPOCHS - 1],
                    testAccuracy!![NUM_EPOCHS - 1],
                ),
            )
            println(
                "%.1f examples/sec".format(
                    trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10.0, 9.0)),
                ),
            )

            val batchNormFirstParams =
                model.block.children
                    .values()[1]
                    .parameters
                    .values()
            println("gamma ${batchNormFirstParams[0].array.reshape(-1)}")
            println("beta ${batchNormFirstParams[1].array.reshape(-1)}")
        }

        private fun setSystemProperties() {
            System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
            System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
            System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
            System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
            System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
            System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")
        }

        private fun prepareDataset(
            usage: Dataset.Usage,
            batchSize: Int,
        ): FashionMnist =
            FashionMnist
                .builder()
                .optUsage(usage)
                .setSampling(batchSize, true)
                .optLimit(getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build()

        private fun prepareModelBlock(): SequentialBlock =
            SequentialBlock()
                .add(
                    Conv2d
                        .builder()
                        .setKernelShape(Shape(5, 5))
                        .setFilters(6)
                        .build(),
                ).add(BatchNormBlock(6, 4))
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(Shape(2, 2), Shape(2, 2)))
                .add(
                    Conv2d
                        .builder()
                        .setKernelShape(Shape(5, 5))
                        .setFilters(16)
                        .build(),
                ).add(BatchNormBlock(16, 4))
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(Shape(2, 2), Shape(2, 2)))
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(120).build())
                .add(BatchNormBlock(120, 2))
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(84).build())
                .add(BatchNormBlock(84, 2))
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(10).build())
    }
}
