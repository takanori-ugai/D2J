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
import ai.djl.training.Trainer
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import jp.live.ugai.d2j.util.Training.trainingChapter6

fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    val net =
        SequentialBlock()
            .add(
                Conv2d.builder()
                    .setKernelShape(Shape(5, 5))
                    .setFilters(6).build(),
            )
            .add(BatchNormBlock(6, 4))
            .add(Pool.maxPool2dBlock(Shape(2, 2), Shape(2, 2)))
            .add(
                Conv2d.builder()
                    .setKernelShape(Shape(5, 5))
                    .setFilters(16).build(),
            )
            .add(BatchNormBlock(16, 4))
            .add(Activation::sigmoid)
            .add(Pool.maxPool2dBlock(Shape(2, 2), Shape(2, 2)))
            .add(Blocks.batchFlattenBlock())
            .add(Linear.builder().setUnits(120).build())
            .add(BatchNormBlock(120, 2))
            .add(Activation::sigmoid)
            .add(Blocks.batchFlattenBlock())
            .add(Linear.builder().setUnits(84).build())
            .add(BatchNormBlock(84, 2))
            .add(Activation::sigmoid)
            .add(Linear.builder().setUnits(10).build())

    val batchSize = 256
    val numEpochs = Integer.getInteger("MAX_EPOCH", 10)

//    var trainLoss: DoubleArray
//    var testAccuracy: DoubleArray
//    val epochCount: DoubleArray
//    var trainAccuracy: DoubleArray

    val epochCount = IntArray(numEpochs) { it + 1 }

    val trainIter =
        FashionMnist.builder()
            .optUsage(Dataset.Usage.TRAIN)
            .setSampling(batchSize, true)
            .optLimit(getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build()

    val testIter =
        FashionMnist.builder()
            .optUsage(Dataset.Usage.TEST)
            .setSampling(batchSize, true)
            .optLimit(getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build()

    trainIter.prepare()
    testIter.prepare()

    val lr = 1.0f

    val loss: Loss = Loss.softmaxCrossEntropyLoss()

    val lrt: Tracker = Tracker.fixed(lr)
    val sgd: Optimizer = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config =
        DefaultTrainingConfig(loss)
            .optOptimizer(sgd) // Optimizer (loss function)
            .optDevices(Engine.getInstance().getDevices(1)) // single GPU
            .addEvaluator(Accuracy()) // Model Accuracy
            .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val model: Model = Model.newInstance("batch-norm")
    model.block = net
    val trainer: Trainer = model.newTrainer(config)
    trainer.initialize(Shape(1, 1, 28, 28))

    val evaluatorMetrics: MutableMap<String, DoubleArray> = mutableMapOf()
    val avgTrainTimePerEpoch = trainingChapter6(trainIter, testIter, numEpochs, trainer, evaluatorMetrics)

    val trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss")
    val trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy")
    val testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy")

    print("loss %.3f,".format(trainLoss!![numEpochs - 1]))
    print(" train acc %.3f,".format(trainAccuracy!![numEpochs - 1]))
    print(" test acc %.3f\n".format(testAccuracy!![numEpochs - 1]))
    print("%.1f examples/sec".format(trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10.0, 9.0))))
    println()

    val batchNormFirstParams = net.children.values()[1].parameters.values()
    println("gamma ${batchNormFirstParams[0].array.reshape(-1)}")
    println("beta ${batchNormFirstParams[1].array.reshape(-1)}")
}

class BatchNormBlock(numFeatures: Int, numDimensions: Int) : AbstractBlock() {
    private var movingMean: NDArray
    private var movingVar: NDArray
    private var gamma: Parameter
    private var beta: Parameter
    private var shape: Shape

    // num_features: the number of outputs for a fully-connected layer
    // or the number of output channels for a convolutional layer.
    // num_dims: 2 for a fully-connected layer and 4 for a convolutional layer.
    init {
        shape =
            if (numDimensions == 2) {
                Shape(1, numFeatures.toLong())
            } else {
                Shape(1, numFeatures.toLong(), 1, 1)
            }
        // The scale parameter and the shift parameter involved in gradient
        // finding and iteration are initialized to 0 and 1 respectively
        gamma =
            addParameter(
                Parameter.builder()
                    .setName("gamma")
                    .setType(Parameter.Type.GAMMA)
                    .optShape(shape)
                    .build(),
            )
        beta =
            addParameter(
                Parameter.builder()
                    .setName("beta")
                    .setType(Parameter.Type.BETA)
                    .optShape(shape)
                    .build(),
            )

        // All the variables not involved in gradient finding and iteration are
        // initialized to 0. Create a base manager to maintain their values
        // throughout the entire training process
        val manager = NDManager.newBaseManager()
        movingMean = manager.zeros(shape)
        movingVar = manager.zeros(shape)
    }

    fun batchNormUpdate(
        X: NDArray,
        gamma: NDArray,
        beta: NDArray,
        movingMean0: NDArray,
        movingVar0: NDArray,
        eps: Float,
        momentum: Float,
        isTraining: Boolean,
    ): NDList {
        // attach moving mean and var to submanager to close intermediate computation values
        // at the end to avoid memory leak
        var movingMean = movingMean0
        var movingVar = movingVar0
        movingMean.manager.newSubManager().use { subManager ->
            movingMean.attach(subManager)
            movingVar.attach(subManager)
            val xHat: NDArray
            val mean: NDArray
            val vari: NDArray
            if (!isTraining) {
                // If it is the prediction mode, directly use the mean and variance
                // obtained from the incoming moving average
                xHat = X.sub(movingMean).div(movingVar.add(eps).sqrt())
            } else {
                if (X.shape.dimension() == 2) {
                    // When using a fully connected layer, calculate the mean and
                    // variance on the feature dimension
                    mean = X.mean(intArrayOf(0), true)
                    vari = X.sub(mean).pow(2).mean(intArrayOf(0), true)
                } else {
                    // When using a two-dimensional convolutional layer, calculate the
                    // mean and variance on the channel dimension (axis=1). Here we
                    // need to maintain the shape of `X`, so that the broadcast
                    // operation can be carried out later
                    mean = X.mean(intArrayOf(0, 2, 3), true)
                    vari = X.sub(mean).pow(2).mean(intArrayOf(0, 2, 3), true)
                }
                // In training mode, the current mean and variance are used for the
                // standardization
                xHat = X.sub(mean).div(vari.add(eps).sqrt())
                // Update the mean and variance of the moving average
                movingMean = movingMean.mul(momentum).add(mean.mul(1.0f - momentum))
                movingVar = movingVar.mul(momentum).add(vari.mul(1.0f - momentum))
            }
            val Y = xHat.mul(gamma).add(beta) // Scale and shift
            // attach moving mean and var back to original manager to keep their values
            movingMean.attach(subManager.parentManager)
            movingVar.attach(subManager.parentManager)
            return NDList(Y, movingMean, movingVar)
        }
    }

    override fun toString(): String {
        return "jp.live.ugai.d2j.BatchNormBlock()"
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
        // close previous NDArray before assigning new values
        if (training) {
            movingMean.close()
            movingVar.close()
        }
        // Save the updated `movingMean` and `movingVar`
        movingMean = result[1]
        movingVar = result[2]
        return NDList(result[0])
    }

    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        var current = inputs
        for (block in children.values()) {
            current = block.getOutputShapes(current)
        }
        return current
    }
}
