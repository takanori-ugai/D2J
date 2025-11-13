package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.Blocks
import ai.djl.nn.SequentialBlock
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.BatchNorm
import ai.djl.nn.pooling.Pool
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.Trainer
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import jp.live.ugai.d2j.util.Training
import jp.live.ugai.d2j.util.getLong

/**
 * The main function where the execution of code starts.
 */
fun main() {
    // Set system properties
    setSystemProperties()

    val batchSize = 256
    val numEpochs = 10

    // Prepare training and testing datasets
    val trainIter = prepareDataset(Dataset.Usage.TRAIN, batchSize)
    val testIter = prepareDataset(Dataset.Usage.TEST, batchSize)

    trainIter.prepare()
    testIter.prepare()

    // Prepare model block
    val block = prepareModelBlock()

    val loss: Loss = Loss.softmaxCrossEntropyLoss()
    val lrt: Tracker = Tracker.fixed(1.0f)
    val sgd: Optimizer = Optimizer.sgd().setLearningRateTracker(lrt).build()

    // Initialize model
    val model: Model = Model.newInstance("batch-norm")
    model.block = block

    // Configure training
    val config =
        DefaultTrainingConfig(loss)
            .optOptimizer(sgd) // Optimizer (loss function)
            .addEvaluator(Accuracy()) // Model Accuracy
            .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    // Initialize trainer
    val trainer: Trainer = model.newTrainer(config)
    trainer.initialize(Shape(1, 1, 28, 28))

    // Train the model
    val evaluatorMetrics: MutableMap<String, DoubleArray> = mutableMapOf()
    val avgTrainTimePerEpoch = Training.trainingChapter6(trainIter, testIter, numEpochs, trainer, evaluatorMetrics)
}

/**
 * Sets the system properties for logging.
 */
fun setSystemProperties() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")
}

/**
 * Prepares the dataset for training or testing.
 *
 * @param usage Dataset usage (TRAIN or TEST).
 * @param batchSize The number of samples per batch.
 * @return The prepared FashionMnist dataset.
 */
fun prepareDataset(
    usage: Dataset.Usage,
    batchSize: Int,
): FashionMnist =
    FashionMnist
        .builder()
        .optUsage(usage)
        .setSampling(batchSize, true)
        .optLimit(getLong("DATASET_LIMIT", Long.MAX_VALUE))
        .build()
        .apply { prepare() }

/**
 * Prepares the model block for the neural network.
 *
 * @return The prepared SequentialBlock.
 */
fun prepareModelBlock(): SequentialBlock =
    SequentialBlock()
        .add(
            Conv2d
                .builder()
                .setKernelShape(Shape(5, 5))
                .setFilters(6)
                .build(),
        ).add(BatchNorm.builder().build())
        .add(Pool.maxPool2dBlock(Shape(2, 2), Shape(2, 2)))
        .add(
            Conv2d
                .builder()
                .setKernelShape(Shape(5, 5))
                .setFilters(16)
                .build(),
        ).add(BatchNorm.builder().build())
        .add(Activation::sigmoid)
        .add(Pool.maxPool2dBlock(Shape(2, 2), Shape(2, 2)))
        .add(Blocks.batchFlattenBlock())
        .add(Linear.builder().setUnits(120).build())
        .add(BatchNorm.builder().build())
        .add(Activation::sigmoid)
        .add(Blocks.batchFlattenBlock())
        .add(Linear.builder().setUnits(84).build())
        .add(BatchNorm.builder().build())
        .add(Activation::sigmoid)
        .add(Linear.builder().setUnits(10).build())

class BatchNorm2
