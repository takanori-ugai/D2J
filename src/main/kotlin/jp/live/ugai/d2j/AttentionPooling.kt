package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import org.jetbrains.letsPlot.geom.geomBin2D
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.geom.geomPoint
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.letsPlot
import org.jetbrains.letsPlot.pos.positionIdentity
import org.jetbrains.letsPlot.scale.scaleFillGradient

fun main() {
    val manager = NDManager.newBaseManager()

    val batchSize = 10
    val n = 50

    val f = { x: NDArray -> x.sin().mul(2).add(x.pow(0.8)) }
    val xTrain = manager.randomUniform(0f, 1f, Shape(n.toLong())).mul(5).sort()
    val yTrain = f(xTrain).add(manager.randomNormal(Shape(n.toLong())))
    val xVal = manager.arange(0f, 5f, 5.0f / n)
    val yVal = f(xVal)
    val nonLinearDataSet = ArrayDataset.Builder()
        .setData(xTrain, xVal)
        .optLabels(yTrain, yVal)
        .setSampling(batchSize, false)
        .build()

    println(xTrain)
    val yHat = yTrain.mean().reshape(1).repeat(n.toLong())
    println(yHat)
    var data = mapOf(
        "x" to xVal.toFloatArray() + xVal.toFloatArray(),
        "y" to yVal.toFloatArray() + yHat.toFloatArray(),
        "label" to Array(n) { "True" } + Array(n) { "Pred" }
    )
    var plot = letsPlot(data)
    plot += geomLine(size = 2) { x = "x" ; y = "y" ; color = "label" }
    plot += geomPoint(size = 3) { x = xTrain.toFloatArray() ; y = yTrain.toFloatArray() }
    plot + ggsize(700, 500)

    fun diff(queries: NDArray, keys: NDArray): NDArray {
        return queries.reshape(-1, 1).sub(keys.reshape(1, -1))
    }

    fun attentionPool(queryKeyDiffs: NDArray, values: NDArray): NDList {
        val attentionWeights = queryKeyDiffs.pow(2).div(2).mul(-1).softmax(1)
        return NDList(attentionWeights.dot(values), attentionWeights)
    }

    val aPool = attentionPool(diff(xVal, xTrain), yTrain)
    data = mapOf(
        "x" to xVal.toFloatArray() + xVal.toFloatArray(),
        "y" to yVal.toFloatArray() + aPool[0].toFloatArray(),
        "label" to Array(n) { "True" } + Array(n) { "Pred" }
    )
    plot = letsPlot(data)
    plot += geomLine(size = 2) { x = "x" ; y = "y" ; color = "label" }
    plot += geomPoint(size = 3) { x = xTrain.toFloatArray() ; y = yTrain.toFloatArray() }
    plot + ggsize(700, 500)

    val matrix = aPool[1]
    val seriesX = mutableListOf<Long>()
    val seriesY = mutableListOf<Long>()
    val seriesW = mutableListOf<Float>()
    for (i in 0 until matrix.shape[0]) {
        val row = matrix.get(i)
        for (j in 0 until row.shape[0]) {
            seriesX.add(j)
            seriesY.add(i)
            seriesW.add(row.get(j).getFloat())
        }
    }
    var data0 = mapOf("x" to seriesX, "y" to seriesY)
    plot = letsPlot(data0)
    plot += geomBin2D(drop = false, binWidth = Pair(1, 1), position = positionIdentity) { x = "x"; y = "y"; weight = seriesW }
    plot += scaleFillGradient(low = "blue", high = "red")
// plot += scaleFillContinuous("red", "green")
    plot + ggsize(700, 200)

    val X = manager.ones(Shape(2, 1, 4))
    val Y = manager.ones(Shape(2, 4, 6))
    println(X.matMul(Y))

    val weights = manager.ones(Shape(2, 10)).mul(0.1)
    val values = manager.arange(20.0f).reshape(Shape(2, 10))
    println(weights.expandDims(1).matMul(values.expandDims(-1)))

    class NWKernelRegression(val keys: NDArray, val values0: NDArray) : AbstractBlock() {
        val w: NDArray = manager.ones(Shape(1))
        var attention: NDArray? = null
        init {
            w.setRequiresGradient(true)
        }

        override fun forwardInternal(
            parameterStore: ParameterStore,
            X: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            val input = X.head()
            val ret = attentionPool(diff(input, keys).mul(w), values0)
            attention = ret[1]
            return ret
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        }

        override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
            return inputs
        }
    }

    val lr = 1f
    val lrt = Tracker.fixed(lr)

    val l2loss = Loss.l2Loss()
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config = DefaultTrainingConfig(l2loss)
        .optOptimizer(sgd) // Optimizer (loss function)
        .optDevices(Engine.getInstance().getDevices(1)) // single CPU/GPU
        .addEvaluator(Accuracy()) // Model Accuracy
        .addEvaluator(l2loss)
        .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val model = Model.newInstance("NWKernelRegression")
    val net = NWKernelRegression(xTrain, yTrain)
    model.setBlock(net)
    val trainer = model.newTrainer(config)
    trainer.initialize(Shape(batchSize.toLong(), 2))
    val numEpochs = 10
    val ps = ParameterStore(manager, false)
    for (epoch in 1..numEpochs) {
        // Iterate over dataset
        var loss: Float = 0f
        for (batch in nonLinearDataSet.getData(manager)) {
            val X = batch.getData().head()
            val y = batch.getLabels().head()

            Engine.getInstance().newGradientCollector().use { gc ->
                val yHat = net.forward(ps, NDList(X), true)
                val l = trainer.loss.evaluate(NDList(yHat.head()), NDList(y))
                gc.backward(l) // gradient calculation
                loss += l.toFloatArray()[0]
            }
            sgd.update("w", net.w, net.w.gradient.div(batchSize))
        }
//    sgd.update("w", net.w, net.w.gradient)
        println("LossValue: $loss ($epoch)")
    }
    println(net.w)
}
class AttentionPooling
