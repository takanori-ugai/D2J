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

/**
 * Represents RNNModel.
 */
class RNNModel(
    private val rnnLayer: RecurrentBlock,
    vocabSize: Int,
) : AbstractBlock() {
    private val dense: Linear
    private val vocabSize: Int

    init {
        this.addChildBlock("rnn", rnnLayer)
        this.vocabSize = vocabSize
        dense = Linear.builder().setUnits(vocabSize.toLong()).build()
        this.addChildBlock("linear", dense)
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
        val tokenIndices = inputs[0].transpose().oneHot(vocabSize)
        inputs[0] = tokenIndices
//        println(inputs)
        val result = rnnLayer.forward(parameterStore, inputs, training)
        val rnnOutput = result[0]
        val state = result[1]
        val shapeLength = rnnOutput.shape.dimension()
        val output =
            dense.forward(
                parameterStore,
                NDList(rnnOutput.reshape(Shape(-1, rnnOutput.shape[shapeLength - 1]))),
                training,
            )
        return NDList(output[0], state)
    }

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        val inputShape = inputShapes[0]
        val rnnInputShape =
            if (inputShape.dimension() == 2) {
                // Input is token indices: (batch, time) -> one-hot -> (time, batch, vocab)
                Shape(inputShape[1], inputShape[0], vocabSize.toLong())
            } else {
                inputShape
            }
        rnnLayer.initialize(manager, dataType, rnnInputShape)
        val rnnOutShape = rnnLayer.getOutputShapes(arrayOf(rnnInputShape))[0]
        dense.initialize(manager, dataType, Shape(1, rnnOutShape.get(rnnOutShape.dimension() - 1)))
    }

    // We won't implement this since we won't be using it but it's required as part of an AbstractBlock

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape?> = arrayOfNulls<Shape>(0)
}
