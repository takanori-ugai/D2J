package jp.live.ugai.d2j

import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.Batch
import ai.djl.training.dataset.Dataset
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.letsPlot

/**
 * Trains and evaluates a softmax regression model on Fashion-MNIST.
 */
fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    class Animator {
        val epochs = mutableListOf<Int>()
        val values = mutableListOf<Float>()
        val metrics = mutableListOf<String>()

        // Add a single metric to the table
        fun add(
            epoch: Int,
            value: Float,
            metric: String,
        ) {
            epochs.add(epoch)
            values.add(value)
            metrics.add(metric)
        }

        // Add accuracy, train accuracy, and train loss metrics for a given epoch
        // Then plot it on the graph
        fun add(
            epoch: Int,
            accuracy: Float,
            trainAcc: Float,
            trainLoss: Float,
        ) {
            add(epoch, trainLoss, "train loss")
            add(epoch, trainAcc, "train accuracy")
            add(epoch, accuracy, "test accuracy")
        }

        // Display the graph
        fun show(): Plot {
            val data = mapOf("epoch" to epochs, "value" to values, "metrics" to metrics)
            // updateDisplay(id, LinePlot.create("", data, "epoch", "value", "metric"));
//        println(data)
            var plot = letsPlot(data)
            plot +=
                geomLine {
                    x = "epoch"
                    y = "value"
                    color = "metric"
                }
            return plot + ggsize(500, 500)
        }
    }

    val batchSize = 256
    val randomShuffle = true

    val numInputs = 784
    val numOutputs = 10
    val manager = NDManager.newBaseManager()
    val weights = manager.randomNormal(0f, 0.01f, Shape(numInputs.toLong(), numOutputs.toLong()), DataType.FLOAT32)
    val bias = manager.zeros(Shape(numOutputs.toLong()), DataType.FLOAT32)
    val params = NDList(weights, bias)

    fun updater(
        params: NDList,
        lr: Float,
        batchSize: Int,
    ) {
//        sgd(params, lr, batchSize)
        for (param in params) {
            // Update param in place.
            // param = param - param.gradient * lr / batchSize
            // val ind = params.indexOf(param)
            // params.rep
            // params.set(ind, param.sub(param.getGradient().mul(lr).div(batchSize)))
            param.subi(param.getGradient().mul(lr).div(batchSize))
        }
    }

    fun softmax(input: NDArray): NDArray {
        val shifted = input.sub(input.max(intArrayOf(1), true))
        val inputExp = shifted.exp()
        val partition = inputExp.sum(intArrayOf(1), true)
//        println(inputExp.div(partition))
        return inputExp.div(partition) // The broadcast mechanism is applied here
    }

    fun net(features: NDArray): NDArray {
        val currentW: NDArray = params.get(0)
        val currentB: NDArray = params.get(1)
        return softmax(features.reshape(Shape(-1, numInputs.toLong())).dot(currentW).add(currentB))
    }

    fun crossEntropy(
        yHat: NDArray,
        y: NDArray,
    ): NDArray {
        // Here, y is not guranteed to be of datatype int or long
        // and in our case we know its a float32.
        // We must first convert it to int or long(here we choose int)
        // before we can use it with NDIndex to "pick" indices.
        // It also takes in a boolean for returning a copy of the existing NDArray
        // but we don't want that so we pass in `false`.
        //     return yHat[NDIndex(":, {}", y.toType(DataType.INT32, false))].log().neg()
        val pickIndex =
            NDIndex()
                .addAllDim(Math.floorMod(-1, yHat.shape.dimension()))
                .addPickDim(y)
        return yHat.get(pickIndex).log().neg()
    }

    fun accuracy(
        yHat: NDArray,
        y: NDArray,
    ): Float {
        // Check size of 1st dimension greater than 1
        // to see if we have multiple samples
        return if (yHat.shape.size(1) > 1) {
            // Argmax gets index of maximum args for given axis 1
            // Convert yHat to same dataType as y (int32)
            // Sum up number of true entries
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

    fun evaluateAccuracy(
        net: (NDArray) -> NDArray,
        dataIterator: Iterable<Batch>,
    ): Float {
        val metric: Accumulator = Accumulator(2) // numCorrectedExamples, numExamples
        for (batch in dataIterator) {
            try {
                val features = batch.data.head()
                val y = batch.labels.head()
                metric.add(floatArrayOf(accuracy(net(features), y), y.size().toFloat()))
            } finally {
                batch.close()
            }
        }
        return metric[0] / metric[1]
    }

    val trainingSet =
        FashionMnist
            .builder()
            .optUsage(Dataset.Usage.TRAIN)
            .setSampling(batchSize, randomShuffle)
            .optLimit(java.lang.Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build()
    val validationSet =
        FashionMnist
            .builder()
            .optUsage(Dataset.Usage.TEST)
            .setSampling(batchSize, false)
            .optLimit(java.lang.Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build()

    var sampleData = manager.create(arrayOf(intArrayOf(1, 2, 3), intArrayOf(4, 5, 6)))
    println(sampleData.sum(intArrayOf(0), true))
    println(sampleData.sum(intArrayOf(1), true))
    println(sampleData.sum(intArrayOf(0, 1), true))

    sampleData = manager.randomNormal(Shape(2, 5))
    val prob = softmax(sampleData)
    println(prob)
    println(prob.sum(intArrayOf(1)))

    val yHat = manager.create(arrayOf(floatArrayOf(0.1f, 0.3f, 0.6f), floatArrayOf(0.3f, 0.2f, 0.5f)))
    val index = NDIndex().addAllDim().addPickDim(manager.create(intArrayOf(0, 2)))
    println(yHat[index])
    val y = manager.create(intArrayOf(0, 2))
    accuracy(yHat, y) / y.size()
    evaluateAccuracy(::net, validationSet.getData(manager))

    val numEpochs = 5
    val lr = 0.1f

    fun trainEpochCh3(
        net: (NDArray) -> NDArray,
        trainIter: Iterable<Batch>,
        loss: (NDArray, NDArray) -> NDArray,
        updater: (NDList, Float, Int) -> Unit,
    ): FloatArray {
        val metric: Accumulator = Accumulator(3) // trainLossSum, trainAccSum, numExamples

        // Attach Gradients
        for (param in params) {
            param.setRequiresGradient(true)
        }
        for (batch in trainIter) {
            var batchFeatures = batch.data.head()
            val y = batch.labels.head()
            batchFeatures = batchFeatures.reshape(Shape(-1, numInputs.toLong()))
            Engine.getInstance().newGradientCollector().use { gc ->
                // Minibatch loss in features and y
                val yHat = net(batchFeatures)
                val l: NDArray = loss(yHat, y)
                gc.backward(l) // Compute gradient on l with respect to w and b
                metric.add(
                    floatArrayOf(
                        l.sum().toType(DataType.FLOAT32, false).getFloat(),
                        accuracy(yHat, y),
                        y.size().toFloat(),
                    ),
                )
                gc.close()
            }
            updater(params, lr, batch.size) // Update parameters using their gradient
            batch.close()
        }
        // Return trainLoss, trainAccuracy
        return floatArrayOf(metric[0] / metric[2], metric[1] / metric[2])
    }

    fun trainCh3(
        net: (NDArray) -> NDArray,
        trainDataset: Dataset,
        testDataset: Dataset,
        loss: (NDArray, NDArray) -> NDArray,
        numEpochs: Int,
        updater: (NDList, Float, Int) -> Unit,
    ) {
        val animator = Animator()
        for (i in 1..numEpochs) {
            val trainMetrics = trainEpochCh3(net, trainDataset.getData(manager), loss, updater)
            val accuracy = evaluateAccuracy(net, testDataset.getData(manager))
            val trainAccuracy = trainMetrics[1]
            val trainLoss = trainMetrics[0]
            animator.add(i, accuracy, trainAccuracy, trainLoss)
            print("Epoch %d: Test Accuracy: %f\n".format(i, accuracy))
            print("Train Accuracy: %f\n".format(trainAccuracy))
            print("Train Loss: %f\n".format(trainLoss))
        }
    }

    trainCh3(::net, trainingSet, validationSet, ::crossEntropy, numEpochs, ::updater)
}

/**
 * Represents Accumulator.
 */
class Accumulator(
    n: Int,
) {
    /**
     * The data.
     */
    var data: FloatArray = FloatArray(n)

    // Adds a set of numbers to the array

    /**
     * Executes add.
     */
    fun add(args: FloatArray) {
        require(args.size == data.size) { "Input array size must match accumulator size." }
        for (i in args.indices) {
            data[i] += args[i]
        }
    }

    // Resets the array

    /**
     * Executes reset.
     */
    fun reset() {
        data.fill(0f)
    }

    // Returns the data point at the given index

    /**
     * Executes get.
     */
    operator fun get(index: Int): Float = data[index]
}

/**
 * Container for softmax regression scratch implementation examples.
 */
internal class SoftRegressionScratch
