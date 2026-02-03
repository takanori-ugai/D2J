package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.ParameterStore
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.Initializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.geom.geomPoint
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.letsPlot
import java.lang.Thread.sleep

/**
 * Implements a generic Attention Pooling block.
 * It takes scores and values, and computes the weighted sum of values based on the softmax of scores.
 */
class AttentionPooling : AbstractBlock() {
    /**
     * The attentionWeights.
     */
    var attentionWeights: NDArray? = null
        private set

    /**
     * Executes forwardInternal.
     */
    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val scores = inputs[0]
        val values = inputs[1]

        val weights = scores.softmax(-1)
        attentionWeights = if (training) null else weights
        val out = weights.matMul(values.reshape(-1, 1)).flatten()
        return NDList(out)
    }

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        // The output shape is (num_queries,)
        return arrayOf(Shape(inputShapes[0][0]))
    }
}

/**
 * Non-parametric Nadaraya-Watson kernel regression model using Attention Pooling.
 */
class NWKernelRegression(
    private val keys: NDArray,
    private val values: NDArray,
) : AbstractBlock() {
    private val attention = addChildBlock("attention", AttentionPooling())
    private val wParam: Parameter =
        addParameter(
            Parameter
                .builder()
                .setName("weight")
                .setType(Parameter.Type.OTHER)
                .optShape(Shape(1))
                .optInitializer(Initializer.ONES) // Use an initializer
                .build(),
        )

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        // The parameter is initialized by the Initializer, so no need to set it here.
        val queriesShape = inputShapes[0]
        val scoresShape = Shape(queriesShape[0], keys.shape[0])
        attention.initialize(manager, dataType, scoresShape, values.shape)
    }

    /**
     * Executes forwardInternal.
     */
    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val queries = inputs.head()
        val keys = if (keys.device != queries.device) keys.toDevice(queries.device, false) else keys
        val values = if (values.device != queries.device) values.toDevice(queries.device, false) else values
        val weight = parameterStore.getValue(wParam, queries.device, training)

        val queryKeyDiffs = queries.reshape(-1, 1).sub(keys.reshape(1, -1))
        // Calculate scores using the Gaussian kernel
        val scores = queryKeyDiffs.mul(weight).pow(2).div(-2.0)

        return attention.forward(parameterStore, NDList(scores, values), training)
    }

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        val scoresShape = Shape(inputs[0][0], keys.shape[0])
        return attention.getOutputShapes(arrayOf(scoresShape, values.shape))
    }
}

/**
 * Main class to run the Nadaraya-Watson kernel regression example.
 */
object AttentionPoolingExample {
    private const val BATCH_SIZE = 10
    private const val NUM_TRAIN = 50
    private const val NUM_VAL = 50
    private const val NUM_EPOCHS = 10
    private const val LEARNING_RATE = 1.0f

    /**
     * Executes main.
     */
    @JvmStatic
    fun main(args: Array<String>) {
        System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
        System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
        System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")

        val manager = NDManager.newBaseManager()

        // --- 1. Generate Data ---
        val f = { x: NDArray -> x.sin().mul(2).add(x.pow(0.8)) }
        val xTrain = manager.randomUniform(0f, 1f, Shape(NUM_TRAIN.toLong())).mul(5).sort()
        val yTrain = f(xTrain).add(manager.randomNormal(Shape(NUM_TRAIN.toLong())))
        val xVal = manager.arange(0f, 5f, 5.0f / NUM_VAL)
        val yVal = f(xVal)

        val trainDataset =
            ArrayDataset
                .Builder()
                .setData(xTrain)
                .optLabels(yTrain)
                .setSampling(BATCH_SIZE, false)
                .build()
        val valDataset =
            ArrayDataset
                .Builder()
                .setData(xVal)
                .optLabels(yVal)
                .setSampling(BATCH_SIZE, false)
                .build()

        // --- 2. Train the Model ---
        val net = NWKernelRegression(xTrain, yTrain)
        val trainer = setupTrainer(net)
        EasyTrain.fit(trainer, NUM_EPOCHS, trainDataset, valDataset)

        // --- 3. Plot Results ---
        val trainedWeight =
            net.parameters[0]
                .value.array
                .getFloat()
        println("Learned weight: $trainedWeight")

        val ps = ParameterStore(manager, false)
        val pred = net.forward(ps, NDList(xVal), false).head()

        plotResults(xVal, yVal, xTrain, yTrain, pred)
        sleep(5)
        println(yVal)
        println(yTrain)
        println(pred)

        val queryBatch = manager.ones(Shape(2, 1, 4))
        val valueBatch = manager.ones(Shape(2, 4, 6))
        println(queryBatch.matMul(valueBatch))

        val weights = manager.ones(Shape(2, 10)).mul(0.1)
        val values = manager.arange(20.0f).reshape(Shape(2, 10))
        println(weights.expandDims(1).matMul(values.expandDims(-1)))
    }

    private fun plotResults(
        xVal: NDArray,
        yVal: NDArray,
        xTrain: NDArray,
        yTrain: NDArray,
        pred: NDArray,
    ) {
        val plotData =
            mapOf(
                "x" to xVal.toFloatArray() + xTrain.toFloatArray() + xVal.toFloatArray(),
                "y" to yVal.toFloatArray() + yTrain.toFloatArray() + pred.toFloatArray(),
                "label" to
                    Array(NUM_VAL) { "True" } +
                    Array(NUM_TRAIN) { "Train" } +
                    Array(NUM_VAL) { "Pred" },
            )
        var plot = letsPlot(plotData)
        plot +=
            geomLine(size = 2) {
                x = "x"
                y = "y"
                color = "label"
            }
        plot +=
            geomPoint(
                data = mapOf("x" to xTrain.toFloatArray(), "y" to yTrain.toFloatArray()),
                size = 2,
            ) {
                x = "x"
                y = "y"
            }
        plot + ggsize(700, 500)
    }

    private fun setupTrainer(net: NWKernelRegression): ai.djl.training.Trainer {
        val l2loss = Loss.l2Loss()
        val config =
            DefaultTrainingConfig(l2loss)
                .optOptimizer(Optimizer.sgd().setLearningRateTracker(Tracker.fixed(LEARNING_RATE)).build())
                .optDevices(Engine.getInstance().getDevices(1))
                .addEvaluator(l2loss)
                .also { cfg ->
                    TrainingListener.Defaults.logging().forEach { cfg.addTrainingListeners(it) }
                }

        val model = Model.newInstance("NWKernelRegression")
        model.block = net
        val trainer = model.newTrainer(config)
        trainer.initialize(Shape(BATCH_SIZE.toLong()))
        trainer.metrics = Metrics()
        return trainer
    }
}
