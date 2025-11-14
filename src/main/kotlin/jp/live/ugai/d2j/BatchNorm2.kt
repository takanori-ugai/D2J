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
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import jp.live.ugai.d2j.util.Training
import jp.live.ugai.d2j.util.getLong

/**
 * Demonstrates training a convolutional neural network with batch normalization on the
 * FashionMNIST dataset using DJL.
 */
class BatchNorm2 {
    companion object {
        private const val BATCH_SIZE = 256 // Number of samples per training batch
        private const val NUM_EPOCHS = 10 // Number of training epochs
        private const val LEARNING_RATE = 0.1f // Learning rate for optimizer

        /**
         * Main entry point for training and evaluating the batch normalization model.
         *
         * Sets up system properties, prepares datasets, configures the model and trainer,
         * and runs the training loop.
         */
        @JvmStatic
        fun main(args: Array<String>) {
            setSystemProperties()

            val trainIter = prepareDataset(Dataset.Usage.TRAIN, BATCH_SIZE)
            val testIter = prepareDataset(Dataset.Usage.TEST, BATCH_SIZE)

            val loss = Loss.softmaxCrossEntropyLoss()
            val tracker = Tracker.fixed(LEARNING_RATE)
            val optimizer = Optimizer.sgd().setLearningRateTracker(tracker).build()

            val model: Model = Model.newInstance("batch-norm")
            model.block = prepareModelBlock()

            val config =
                DefaultTrainingConfig(loss)
                    .optOptimizer(optimizer)
                    .addEvaluator(Accuracy())
                    .addTrainingListeners(*TrainingListener.Defaults.logging())

            val trainer = model.newTrainer(config)
            trainer.initialize(Shape(1, 1, 28, 28))

            val evaluatorMetrics: MutableMap<String, DoubleArray> = mutableMapOf()
            Training.trainingChapter6(trainIter, testIter, NUM_EPOCHS, trainer, evaluatorMetrics)
        }

        /**
         * Sets system properties for logging configuration.
         */
        private fun setSystemProperties() {
            System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
            System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
            System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
            System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
            System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
            System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")
        }

        /**
         * Prepares the FashionMNIST dataset for the given usage and batch size.
         *
         * @param usage The dataset usage type (TRAIN or TEST).
         * @param batchSize The number of samples per batch.
         * @return The prepared FashionMnist dataset.
         */
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

        /**
         * Builds the model block with convolutional, batch normalization, activation,
         * pooling, and linear layers.
         *
         * @return The constructed SequentialBlock for the model.
         */
        private fun prepareModelBlock() =
            SequentialBlock()
                .add(
                    Conv2d
                        .builder()
                        .setKernelShape(Shape(5, 5))
                        .setFilters(6)
                        .build(),
                ).add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(Shape(2, 2), Shape(2, 2)))
                .add(
                    Conv2d
                        .builder()
                        .setKernelShape(Shape(5, 5))
                        .setFilters(16)
                        .build(),
                ).add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(Shape(2, 2), Shape(2, 2)))
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(120).build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Linear.builder().setUnits(84).build())
                .add(BatchNorm.builder().build())
                .add(Activation::relu)
                .add(Linear.builder().setUnits(10).build())
    }
}
