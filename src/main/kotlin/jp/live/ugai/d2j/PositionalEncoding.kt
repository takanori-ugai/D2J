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

class PositionalEncoding(numHiddens: Int, dropout: Float, maxLen: Int, manager: NDManager) : AbstractBlock() {
    private val dropout: Dropout
    var P: NDArray

    init {
        this.dropout = Dropout.builder().optRate(dropout).build()
        addChildBlock("dropout", this.dropout)

        // Create a long enough `P`
        P = manager.zeros(Shape(1, maxLen.toLong(), numHiddens.toLong()))
        val X = manager.arange(maxLen)
            .reshape(-1, 1)
            .div(
                manager.create(10000)
                    .pow(manager.arange(0, numHiddens, 2).div(numHiddens))
            )
        P[NDIndex(":, :, {}::{}", 0, 2)] = X.sin()
        P[NDIndex(":, :, {}::{}", 1, 2)] = X.cos()
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var X = inputs[0]
        X = X.add(P[":, :{}, :", X.shape[1]])
        return NDList(dropout.forward(parameterStore, NDList(X), training, params)[0])
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }

    public override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        manager.newSubManager().use { sub ->
            var X = sub.zeros(inputShapes[0], dataType)
            X = X.add(P[":, :{}, :", X.shape[1]])
            dropout.initialize(manager, dataType, X.shape)
        }
    }
}
