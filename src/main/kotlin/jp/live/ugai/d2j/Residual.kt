package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.modality.cv.transform.Resize
import ai.djl.modality.cv.transform.ToTensor
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.Block
import ai.djl.nn.Blocks
import ai.djl.nn.ParallelBlock
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
import jp.live.ugai.d2j.util.Training.trainingChapter6

fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    val manager = NDManager.newBaseManager()

    var blk = SequentialBlock()
    blk.add(Residual(3, false, Shape(1, 1)))

    var X = manager.randomUniform(0f, 1.0f, Shape(4, 3, 6, 6))

    val parameterStore = ParameterStore(manager, true)

    blk.initialize(manager, DataType.FLOAT32, X.shape)

    println(blk.forward(parameterStore, NDList(X), false).singletonOrThrow().shape)

    blk = SequentialBlock()
    blk.add(Residual(6, true, Shape(2, 2)))

    blk.initialize(manager, DataType.FLOAT32, X.shape)

    println(blk.forward(parameterStore, NDList(X), false).singletonOrThrow().shape)

    val net = SequentialBlock()
    net
        .add(
            Conv2d.builder()
                .setKernelShape(Shape(7, 7))
                .optStride(Shape(2, 2))
                .optPadding(Shape(3, 3))
                .setFilters(64)
                .build()
        )
        .add(BatchNorm.builder().build())
        .add(Activation::relu)
        .add(
            Pool.maxPool2dBlock(Shape(3, 3), Shape(2, 2), Shape(1, 1))
        )
    net
        .add(resnetBlock(64, 2, true))
        .add(resnetBlock(128, 2, false))
        .add(resnetBlock(256, 2, false))
        .add(resnetBlock(512, 2, false))
    net
        .add(Pool.globalAvgPool2dBlock())
        .add(Linear.builder().setUnits(10).build())

    X = manager.randomUniform(0f, 1f, Shape(1, 1, 224, 224))
    net.initialize(manager, DataType.FLOAT32, X.shape)
    var currentShape = X.shape

    for (i in 0 until net.children.size()) {
        X = net.children[i].value.forward(parameterStore, NDList(X), false).singletonOrThrow()
        currentShape = X.shape
        println(net.children[i].key + " layer output : " + currentShape)
    }

    val batchSize = 256
    val lr = 0.05f
    val numEpochs = Integer.getInteger("MAX_EPOCH", 10)

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

    val model: Model = Model.newInstance("cnn")
    model.block = net

    val loss: Loss = Loss.softmaxCrossEntropyLoss()

    val lrt: Tracker = Tracker.fixed(lr)
    val sgd: Optimizer = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config = DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
        .addEvaluator(Accuracy()) // Model Accuracy
        .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val trainer: Trainer = model.newTrainer(config)

    val evaluatorMetrics: MutableMap<String, DoubleArray> = mutableMapOf()
    val avgTrainTimePerEpoch = trainingChapter6(trainIter, testIter, numEpochs, trainer, evaluatorMetrics)

    val trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss")
    val trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy")
    val testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy")

    print("loss %.3f,".format(trainLoss!![numEpochs - 1]))
    print(" train acc %.3f,".format(trainAccuracy!![numEpochs - 1]))
    print(" test acc %.3f\n".format(testAccuracy!![numEpochs - 1]))
    println("%.1f examples/sec".format(trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10.0, 9.0))))
}

fun resnetBlock(numChannels: Int, numResiduals: Int, firstBlock: Boolean): SequentialBlock {
    val blk = SequentialBlock()
    for (i in 0 until numResiduals) {
        if (i == 0 && !firstBlock) {
            blk.add(Residual(numChannels, true, Shape(2, 2)))
        } else {
            blk.add(Residual(numChannels, false, Shape(1, 1)))
        }
    }
    return blk
}

class Residual(numChannels: Int, use1x1Conv: Boolean, strideShape: Shape) :
    AbstractBlock(VERSION) {
    var block: ParallelBlock

    init {
        val b1: Block
        val conv1x1: Block
        b1 = SequentialBlock()
        b1.add(
            Conv2d.builder()
                .setFilters(numChannels)
                .setKernelShape(Shape(3, 3))
                .optPadding(Shape(1, 1))
                .optStride(strideShape)
                .build()
        )
            .add(BatchNorm.builder().build())
            .add(Activation::relu)
            .add(
                Conv2d.builder()
                    .setFilters(numChannels)
                    .setKernelShape(Shape(3, 3))
                    .optPadding(Shape(1, 1))
                    .build()
            )
            .add(BatchNorm.builder().build())
        if (use1x1Conv) {
            conv1x1 = SequentialBlock()
            conv1x1.add(
                Conv2d.builder()
                    .setFilters(numChannels)
                    .setKernelShape(Shape(1, 1))
                    .optStride(strideShape)
                    .build()
            )
        } else {
            conv1x1 = SequentialBlock()
            conv1x1.add(Blocks.identityBlock())
        }
        block = addChildBlock(
            "residualBlock",
            ParallelBlock(
                { list: List<NDList> ->
                    val unit = list[0]
                    val parallel = list[1]
                    NDList(
                        unit.singletonOrThrow()
                            .add(parallel.singletonOrThrow())
                            .ndArrayInternal
                            .relu()
                    )
                },
                mutableListOf(b1 as Block, conv1x1)
            )
        )
    }

    override fun toString(): String {
        return "Residual()"
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        return block.forward(parameterStore, inputs, training)
    }

    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        var current: Array<Shape> = inputs
        for (block in block.children.values()) {
            current = block.getOutputShapes(current)
        }
        return current
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        block.initialize(manager, dataType, *inputShapes)
    }

    companion object {
        private const val VERSION: Byte = 2
    }
}
