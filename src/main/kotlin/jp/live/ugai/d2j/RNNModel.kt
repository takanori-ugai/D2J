package jp.live.ugai.d2j

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.recurrent.RecurrentBlock
import ai.djl.training.ParameterStore
import ai.djl.util.PairList

class RNNModel(private val rnnLayer: RecurrentBlock, vocabSize: Int) : AbstractBlock() {
    private val dense: Linear
    private val vocabSize: Int

    init {
        this.addChildBlock("rnn", rnnLayer)
        this.vocabSize = vocabSize
        dense = Linear.builder().setUnits(vocabSize.toLong()).build()
        this.addChildBlock("linear", dense)
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        val X = inputs[0].transpose().oneHot(vocabSize)
        inputs[0] = X
//        println(inputs)
        val result = rnnLayer.forward(parameterStore, inputs, training)
        val Y = result[0]
        val state = result[1]
        val shapeLength = Y.shape.dimension()
        val output = dense.forward(
            parameterStore,
            NDList(Y.reshape(Shape(-1, Y.shape[shapeLength - 1]))),
            training
        )
        return NDList(output[0], state)
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        val shape: Shape = rnnLayer.getOutputShapes(arrayOf(inputShapes[0]))[0]
        dense.initialize(manager, dataType, Shape(vocabSize.toLong(), shape.get(shape.dimension() - 1)))
    }

    /* We won't implement this since we won't be using it but it's required as part of an AbstractBlock  */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape?> {
        return arrayOfNulls<Shape>(0)
    }
}
