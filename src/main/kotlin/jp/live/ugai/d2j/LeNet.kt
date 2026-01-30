package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.engine.Engine
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.Blocks
import ai.djl.nn.SequentialBlock
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.core.Linear
import ai.djl.nn.pooling.Pool
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker

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

    Engine.getInstance().setRandomSeed(1111)

    val manager = NDManager.newBaseManager()

    val block = SequentialBlock()
    block
        .add(
            Conv2d
                .builder()
                .setKernelShape(Shape(5, 5))
                .optPadding(Shape(2, 2))
                .optBias(false)
                .setFilters(6)
                .build(),
        ).add(Activation::sigmoid)
        .add(Pool.avgPool2dBlock(Shape(5, 5), Shape(2, 2), Shape(2, 2)))
        .add(
            Conv2d
                .builder()
                .setKernelShape(Shape(5, 5))
                .setFilters(16)
                .build(),
        ).add(Activation::sigmoid)
        .add(Pool.avgPool2dBlock(Shape(5, 5), Shape(2, 2), Shape(2, 2)))
        // Blocks.batchFlattenBlock() will transform the input of the shape (batch size, channel,
        // height, width) into the input of the shape (batch size,
        // channel * height * width)
        .add(Blocks.batchFlattenBlock())
        .add(
            Linear
                .builder()
                .setUnits(120)
                .build(),
        ).add(Activation::sigmoid)
        .add(
            Linear
                .builder()
                .setUnits(84)
                .build(),
        ).add(Activation::sigmoid)
        .add(
            Linear
                .builder()
                .setUnits(10)
                .build(),
        )

    val lr = 0.9f
    val model = Model.newInstance("cnn")
    model.block = block

    val loss = Loss.softmaxCrossEntropyLoss()

    val lrt = Tracker.fixed(lr)
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config =
        DefaultTrainingConfig(loss)
            .optOptimizer(sgd) // Optimizer (loss function)
            .optDevices(Engine.getInstance().getDevices(1)) // Single GPU
            .addEvaluator(Accuracy()) // Model Accuracy
            .also { cfg ->
                TrainingListener.Defaults.basic().forEach { cfg.addTrainingListeners(it) }
            }

    val trainer = model.newTrainer(config)

    val sampleInput = manager.randomUniform(0f, 1.0f, Shape(1, 1, 28, 28))
    trainer.initialize(sampleInput.shape)

    var currentShape = sampleInput.shape

    for (i in 0 until block.children.size()) {
        val newShape =
            block.children
                .get(i)
                .value
                .getOutputShapes(arrayOf<Shape>(currentShape))
        currentShape = newShape[0]
        println(block.children.get(i).key + " layer output : " + currentShape)
    }

    val batchSize = 256
    val numEpochs = 10
//    val epochCount = DoubleArray(numEpochs) { it.toDouble() + 1f }

    val trainIter =
        FashionMnist
            .builder()
            .optUsage(Dataset.Usage.TRAIN)
            .setSampling(batchSize, true)
            .optLimit(Long.MAX_VALUE)
            .build()

    val testIter =
        FashionMnist
            .builder()
            .optUsage(Dataset.Usage.TEST)
            .setSampling(batchSize, true)
            .optLimit(Long.MAX_VALUE)
            .build()

    trainIter.prepare()
    testIter.prepare()

    var trainLoss: DoubleArray?
    var trainAccuracy: DoubleArray? = null
    var testAccuracy: DoubleArray?

    fun trainingChapter6(
        trainIter: ArrayDataset,
        testIter: ArrayDataset,
        numEpochs: Int,
        trainer: Trainer,
    ) {
        var avgTrainTimePerEpoch = 0.0
//    val evaluatorMetrics = new HashMap<>();
        val evaluatorMetrics = mutableMapOf<String, DoubleArray>()

        trainer.metrics = Metrics()

        EasyTrain.fit(trainer, numEpochs, trainIter, testIter)

        val metrics = trainer.metrics

        trainer.evaluators
            .stream()
            .forEach { evaluator ->
                evaluatorMetrics.put(
                    "train_epoch_" + evaluator.name,
                    metrics.getMetric("train_epoch_" + evaluator.name).map { it.value }.toDoubleArray(),
                )
                evaluatorMetrics.put(
                    "validate_epoch_" + evaluator.name,
                    metrics.getMetric("validate_epoch_" + evaluator.name).map { it.value }.toDoubleArray(),
                )
            }

        avgTrainTimePerEpoch = metrics.mean("epoch")

        trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss")
        trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy")
        testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy")

        print("loss %.3f,".format(trainLoss!![numEpochs - 1]))
        print(" train acc %.3f,".format(trainAccuracy!![numEpochs - 1]))
        print(" test acc %.3f\n".format(testAccuracy!![numEpochs - 1]))
        print("%.1f examples/sec \n".format(trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10.0, 9.0))))
    }

    trainingChapter6(trainIter, testIter, numEpochs, trainer)
    println(trainAccuracy!!.toList())
}

/**
 * Represents LeNet.
 */
class LeNet
