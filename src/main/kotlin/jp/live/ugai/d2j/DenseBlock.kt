package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.modality.cv.transform.Resize
import ai.djl.modality.cv.transform.ToTensor
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
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import jp.live.ugai.d2j.util.Training.trainingChapter6

/**
 * Executes main.
 */
fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    val manager = NDManager.newBaseManager()
    var block = SequentialBlock().add(DenseBlock(2, 10))

    val inputTensor = manager.randomUniform(0.0f, 1.0f, Shape(4, 3, 8, 8))

    block.initialize(manager, DataType.FLOAT32, inputTensor.shape)

//    val parameterStore = ParameterStore(manager, true)

    var currentShape = arrayOf(inputTensor.shape)
    for (child in block.children.values()) {
        currentShape = child.getOutputShapes(currentShape)
    }

    println(currentShape[0])

    block = transitionBlock(10)

    block.initialize(manager, DataType.FLOAT32, currentShape[0])

    for (pair in block.children) {
        currentShape = pair.value.getOutputShapes(currentShape)
    }

    println(currentShape[0])

    val net =
        SequentialBlock()
            .add(
                Conv2d
                    .builder()
                    .setFilters(64)
                    .setKernelShape(Shape(7, 7))
                    .optStride(Shape(2, 2))
                    .optPadding(Shape(3, 3))
                    .build(),
            ).add(BatchNorm.builder().build())
            .add { arrays: NDList? -> Activation.relu(arrays) }
            .add(Pool.maxPool2dBlock(Shape(3, 3), Shape(2, 2), Shape(1, 1)))

    var numChannels = 64
    val growthRate = 32

    val numConvsInDenseBlocks = intArrayOf(4, 4, 4, 4)

    for (index in numConvsInDenseBlocks.indices) {
        val numConvs = numConvsInDenseBlocks[index]
        net.add(DenseBlock(numConvs, growthRate))
        numChannels = numChannels + numConvs * growthRate
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

//    var trainLoss: DoubleArray
//    var testAccuracy: DoubleArray
//    var trainAccuracy: DoubleArray

    val trainIter =
        FashionMnist
            .builder()
            .addTransform(Resize(96))
            .addTransform(ToTensor())
            .optUsage(Dataset.Usage.TRAIN)
            .setSampling(batchSize, true)
            .optLimit(java.lang.Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build()

    val testIter =
        FashionMnist
            .builder()
            .addTransform(Resize(96))
            .addTransform(ToTensor())
            .optUsage(Dataset.Usage.TEST)
            .setSampling(batchSize, true)
            .optLimit(java.lang.Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build()

    trainIter.prepare()
    testIter.prepare()

    val model = Model.newInstance("cnn")
    model.block = net

    val loss = Loss.softmaxCrossEntropyLoss()

    val lrt = Tracker.fixed(lr)
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config =
        DefaultTrainingConfig(loss)
            .optOptimizer(sgd) // Optimizer (loss function)
            .addEvaluator(Accuracy()) // Model Accuracy
            .also { cfg ->
                TrainingListener.Defaults.logging().forEach { cfg.addTrainingListeners(it) }
            } // Logging

    val trainer = model.newTrainer(config)
    trainer.initialize(Shape(1, 1, 96, 96))

    val evaluatorMetrics: MutableMap<String, DoubleArray> = mutableMapOf()
    trainingChapter6(trainIter, testIter, numEpochs, trainer, evaluatorMetrics)
}

/**
 * Executes transitionBlock.
 */
fun transitionBlock(numChannels: Int): SequentialBlock =
    SequentialBlock()
        .add(BatchNorm.builder().build())
        .add { arrays: NDList -> Activation.relu(arrays) }
        .add(
            Conv2d
                .builder()
                .setFilters(numChannels)
                .setKernelShape(Shape(1, 1))
                .optStride(Shape(1, 1))
                .build(),
        ).add(Pool.avgPool2dBlock(Shape(2, 2), Shape(2, 2)))

/**
 * A DenseBlock is a series of convolutional blocks that are densely connected. Each block's input
 * includes the outputs of all preceding blocks, promoting feature reuse.
 *
 * @property numConvs The number of convolutional blocks within this dense block.
 * @property numChannels The number of output channels for each convolutional block.
 */
class DenseBlock(
    numConvs: Int,
    numChannels: Int,
) : AbstractBlock(VERSION) {
    /**
     * The net.
     */
    val net = SequentialBlock()

    init {
        for (i in 0 until numConvs) {
            net.add(addChildBlock("denseBlock$i", convBlock(numChannels)))
        }
    }

    /**
     * Returns a string representation of the DenseBlock.
     */
    override fun toString(): String = "DenseBlock()"

    /**
     * Performs a forward pass through all convolutional blocks in the DenseBlock.
     *
     * @param parameterStore Stores parameters for the model during training.
     * @param X The input NDList to the block.
     * @param training A boolean indicating if the model is in training mode.
     * @param params Additional parameters for the forward pass.
     * @return The output NDList after processing through the DenseBlock.
     */
    override fun forwardInternal(
        parameterStore: ParameterStore,
        X: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList =
        net.children.values().fold(X) { acc, block ->
            NDList(
                NDArrays.concat(
                    NDList(
                        acc.singletonOrThrow(),
                        block.forward(parameterStore, acc, training, params).singletonOrThrow(),
                    ),
                    1,
                ),
            )
        }

    /**
     * Calculates the output shapes given the input shapes to the DenseBlock.
     *
     * @param inputs An array of input shapes.
     * @return An array of output shapes after processing through the DenseBlock.
     */
    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        net.children.values().forEach { block ->
            val shapeY = block.getOutputShapes(inputs)[0]
            inputs[0] =
                Shape(
                    inputs[0][0],
                    shapeY[1] + inputs[0][1],
                    inputs[0][2],
                    inputs[0][3],
                )
        }
        return inputs
    }

    /**
     * Initializes child blocks with the specified data type and input shapes.
     *
     * @param manager The NDManager to manage resources.
     * @param dataType The data type for the blocks.
     * @param inputShapes The input shapes to initialize the blocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        var shapesX: Shape = inputShapes[0]
        for (block in net.children.values()) {
            block.initialize(manager, dataType, shapesX)
            val shapesY: Array<Shape> = block.getOutputShapes(arrayOf(shapesX))
            shapesX =
                Shape(
                    shapesX.get(0),
                    shapesY[0].get(1) + shapesX.get(1),
                    shapesX.get(2),
                    shapesX.get(3),
                )
        }
    }

    companion object {
        private const val VERSION: Byte = 1
    }

    /**
     * Creates a convolutional block consisting of BatchNorm, ReLU activation, and Conv2d layers.
     *
     * @param numChannels The number of output channels for the Conv2d layer.
     * @return A SequentialBlock representing the convolutional block.
     */
    private fun convBlock(numChannels: Int): SequentialBlock =
        SequentialBlock()
            .add(BatchNorm.builder().build())
            .add(Activation::relu)
            .add(
                Conv2d
                    .builder()
                    .setFilters(numChannels)
                    .setKernelShape(Shape(3, 3))
                    .optPadding(Shape(1, 1))
                    .optStride(Shape(1, 1))
                    .build(),
            )
}
