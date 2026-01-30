package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.norm.Dropout
import ai.djl.training.ParameterStore
import ai.djl.util.PairList

/**
 * Represents PositionalEncoding.
 */
class PositionalEncoding(
    numHiddens: Int,
    dropout: Float,
    maxLen: Int,
    manager: NDManager,
) : AbstractBlock() {
    private val dropout: Dropout

    /**
     * The posEncoding.
     */
    var posEncoding: NDArray

    init {
        this.dropout = Dropout.builder().optRate(dropout).build()
        addChildBlock("dropout", this.dropout)

        // Create a long enough `P`
        posEncoding = manager.zeros(Shape(1, maxLen.toLong(), numHiddens.toLong()))
        /**
         * The positions.
         */
        val positions =
            manager
                .arange(maxLen)
                .reshape(-1, 1)
                .div(
                    manager
                        .create(10000)
                        .pow(manager.arange(0, numHiddens, 2).div(numHiddens)),
                )
        posEncoding[NDIndex(":, :, {}::{}", 0, 2)] = positions.sin()
        posEncoding[NDIndex(":, :, {}::{}", 1, 2)] = positions.cos()
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
        var input = inputs[0]
        val pos = if (posEncoding.device != input.device) posEncoding.toDevice(input.device, false) else posEncoding
        input = input.add(pos[":, :{}, :", input.shape[1]])
        return NDList(dropout.forward(parameterStore, NDList(input), training, params)[0])
    }

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> =
        throw UnsupportedOperationException("Not implemented")

    /**
     * Executes initializeChildBlocks.
     */
    public override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        manager.newSubManager().use { sub ->
            var input = sub.zeros(inputShapes[0], dataType)
            val pos =
                if (posEncoding.device != manager.device) {
                    posEncoding.toDevice(manager.device, false)
                } else {
                    posEncoding
                }
            input = input.add(pos[":, :{}, :", input.shape[1]])
            dropout.initialize(manager, dataType, input.shape)
        }
    }
}
