package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.modality.cv.transform.Resize
import ai.djl.modality.cv.transform.ToTensor
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.BatchNorm
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
import jp.live.ugai.d2j.Training.trainingChapter6

fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    val manager = NDManager.newBaseManager()
    var block = SequentialBlock().add(DenseBlock(2, 10))

    val X = manager.randomUniform(0.0f, 1.0f, Shape(4, 3, 8, 8))

    block.initialize(manager, DataType.FLOAT32, X.getShape())

    val parameterStore = ParameterStore(manager, true)

    var currentShape = arrayOf<Shape>(X.getShape())
    for (child in block.getChildren().values()) {
        currentShape = child.getOutputShapes(currentShape)
    }

    println(currentShape[0])

    block = transitionBlock(10)

    block.initialize(manager, DataType.FLOAT32, currentShape[0])

    for (pair in block.getChildren()) {
        currentShape = pair.getValue().getOutputShapes(currentShape)
    }

    println(currentShape[0])

    val net = SequentialBlock()
        .add(
            Conv2d.builder()
                .setFilters(64)
                .setKernelShape(Shape(7, 7))
                .optStride(Shape(2, 2))
                .optPadding(Shape(3, 3))
                .build()
        )
        .add(BatchNorm.builder().build())
        .add { arrays: NDList? -> Activation.relu(arrays) }
        .add(Pool.maxPool2dBlock(Shape(3, 3), Shape(2, 2), Shape(1, 1)))

    var numChannels: Int = 64
    val growthRate = 32

    val numConvsInDenseBlocks = intArrayOf(4, 4, 4, 4)

    for (index in numConvsInDenseBlocks.indices) {
        val numConvs = numConvsInDenseBlocks[index]
        net.add(DenseBlock(numConvs, growthRate))
        numChannels += numConvs * growthRate
        if (index != numConvsInDenseBlocks.size - 1) {
            numChannels = numChannels / 2
            net.add(transitionBlock(numChannels))
        }
    }
    net
        .add(BatchNorm.builder().build())
        .add(Activation::relu)
        .add(Pool.globalAvgPool2dBlock())
        .add(Linear.builder().setUnits(10).build())

    println(net)

    val batchSize = 256
    val lr = 0.1f
    val numEpochs = Integer.getInteger("MAX_EPOCH", 10)

    var trainLoss: DoubleArray
    var testAccuracy: DoubleArray
    var trainAccuracy: DoubleArray

    val epochCount = IntArray(numEpochs) { it + 1 }

    val trainIter = FashionMnist.builder()
        .addTransform(Resize(96))
        .addTransform(ToTensor())
        .optUsage(Dataset.Usage.TRAIN)
        .setSampling(batchSize, true)
        .optLimit(java.lang.Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
        .build()

    val testIter = FashionMnist.builder()
        .addTransform(Resize(96))
        .addTransform(ToTensor())
        .optUsage(Dataset.Usage.TEST)
        .setSampling(batchSize, true)
        .optLimit(java.lang.Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
        .build()

    trainIter.prepare()
    testIter.prepare()

    val model = Model.newInstance("cnn")
    model.setBlock(net)

    val loss: Loss = Loss.softmaxCrossEntropyLoss()

    val lrt: Tracker = Tracker.fixed(lr)
    val sgd: Optimizer = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config = DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
        .addEvaluator(Accuracy()) // Model Accuracy
        .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val trainer: Trainer = model.newTrainer(config)
    trainer.initialize(Shape(1, 1, 96, 96))

    val evaluatorMetrics: MutableMap<String, DoubleArray> = mutableMapOf()
    val avgTrainTimePerEpoch = trainingChapter6(trainIter, testIter, numEpochs, trainer, evaluatorMetrics)
}

fun transitionBlock(numChannels: Int): SequentialBlock? {
    return SequentialBlock()
        .add(BatchNorm.builder().build())
        .add { arrays: NDList? -> Activation.relu(arrays) }
        .add(
            Conv2d.builder()
                .setFilters(numChannels)
                .setKernelShape(Shape(1, 1))
                .optStride(Shape(1, 1))
                .build()
        )
        .add(Pool.avgPool2dBlock(Shape(2, 2), Shape(2, 2)))
}

fun convBlock(numChannels: Int): SequentialBlock? {
    return SequentialBlock()
        .add(BatchNorm.builder().build())
        .add(Activation::relu)
        .add(
            Conv2d.builder()
                .setFilters(numChannels)
                .setKernelShape(Shape(3, 3))
                .optPadding(Shape(1, 1))
                .optStride(Shape(1, 1))
                .build()
        )
}

class DenseBlock(numConvs: Int, numChannels: Int) : AbstractBlock(VERSION) {
    var net = SequentialBlock()

    init {
        for (i in 0 until numConvs) {
            net.add(addChildBlock("denseBlock$i", convBlock(numChannels)))
        }
    }

    override fun toString(): String {
        return "DenseBlock()"
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        X: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var X = X
        var Y: NDArray
        for (block in net.children.values()) {
            Y = block.forward(parameterStore, X, training).singletonOrThrow()
            X = NDList(NDArrays.concat(NDList(X.singletonOrThrow(), Y), 1))
        }
        return X
    }

    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        val shapesX: Array<Shape> = inputs
        for (block in net.children.values()) {
            val shapesY: Array<Shape> = block.getOutputShapes(shapesX)
            shapesX[0] = Shape(
                shapesX[0].get(0),
                shapesY[0].get(1) + shapesX[0].get(1),
                shapesX[0].get(2),
                shapesX[0].get(3)
            )
        }
        return shapesX
    }

    override fun initializeChildBlocks(manager: NDManager?, dataType: DataType?, vararg inputShapes: Shape) {
        var shapesX: Shape = inputShapes[0]
        for (block in net.children.values()) {
            block.initialize(manager, DataType.FLOAT32, shapesX)
            val shapesY: Array<Shape> = block.getOutputShapes(arrayOf(shapesX))
            shapesX = Shape(
                shapesX.get(0),
                shapesY[0].get(1) + shapesX.get(1),
                shapesX.get(2),
                shapesX.get(3)
            )
        }
    }

    companion object {
        private const val VERSION: Byte = 1
    }
}
