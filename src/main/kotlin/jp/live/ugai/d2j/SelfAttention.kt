package jp.live.ugai.d2j

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import jp.live.ugai.d2j.attention.MultiHeadAttention

fun main() {
    val manager = NDManager.newBaseManager()

    val numHiddens = 100
    val numHeads = 5
    val attention = MultiHeadAttention(numHiddens, numHeads, 0.5f, false)

    val batchSize = 2
    val numQueries = 4
    val validLens = manager.create(floatArrayOf(3.0f, 2.0f))
    var X = manager.ones(Shape(batchSize.toLong(), numQueries.toLong(), numHiddens.toLong()))
    val ps = ParameterStore(manager, false)
    var input = NDList(X, X, X, validLens)
    attention.initialize(manager, DataType.FLOAT32, *input.shapes)
    val result = attention.forward(ps, input, false)
    println(result.get(0).shape)

    val encodingDim = 32
    val numSteps = 60
    val posEncoding = PositionalEncoding(encodingDim, 0f, 1000, manager)
    input = NDList(manager.zeros(Shape(1, numSteps.toLong(), encodingDim.toLong())))
    X = posEncoding.forward(ps, input, false)[0]
    val P = posEncoding.P[NDIndex(":, :{}, :", X.shape[1])]
    println(P)

    val plotSize = 4
    val plotX = mutableListOf<FloatArray>()
    val plotY = mutableListOf<FloatArray>()
    for (i in 0..3) {
        if (i == 0) {
            plotX.add(manager.arange(numSteps).toType(DataType.FLOAT32, false).toFloatArray())
        } else {
            plotX.add(plotX[i - 1])
        }
        plotY.add(P[NDIndex("0, :, {},", i + 6)].toFloatArray())
    }
    println(plotX[0].toList())
    println(plotY[0].toList())

    for (i in 0..7) {
        println(i.toString() + " in binary is " + Integer.toBinaryString(i))
    }
}

class SelfAttention
