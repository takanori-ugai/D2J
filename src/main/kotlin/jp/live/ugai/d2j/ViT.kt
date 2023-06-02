package jp.live.ugai.d2j

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.Dropout
import ai.djl.nn.norm.LayerNorm
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.ConstantInitializer
import ai.djl.util.PairList

class ViT(
    val imgSize: Int,
    patchSize: Int,
    val numHiddens: Int,
    mlpNumHiddens: Int,
    numHeads: Int,
    numBlks: Int,
    embDropout: Float,
    blkDropout: Float,
    lr: Float = 0.1f,
    useBias: Boolean = false,
    val numClasses: Int = 10
) : AbstractBlock() {
    val patchEmbedding = PatchEmbedding(imgSize, patchSize, numHiddens)
    val clsToken = Parameter.builder()
        .optRequiresGrad(true)
        .setType(Parameter.Type.BIAS)
        .optShape(Shape(1, 1, numHiddens.toLong()))
        .build()
    val numSteps: Int = patchEmbedding.numPatches + 1
    val posEmbedding = Parameter.builder()
        .optRequiresGrad(true)
        .optShape(Shape(1, numSteps.toLong(), numHiddens.toLong()))
//    torch.randn(1, num_steps, num_hiddens))
        .setType(Parameter.Type.BIAS)
        .build()
    val blks0 = SequentialBlock()
        .add(Dropout.builder().optRate(embDropout).build())
    val head = SequentialBlock()
        .add(LayerNorm.builder().build())
        .add(Linear.builder().setUnits(numClasses.toLong()).build())
    init {
        addParameter(clsToken)
        addParameter(posEmbedding)
        clsToken.setInitializer(ConstantInitializer(0f))
        repeat(numBlks) {
            blks0.add(ViTBlock(numHiddens, numHiddens, mlpNumHiddens, numHeads, blkDropout, useBias))
        }
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var X = patchEmbedding.forward(parameterStore, inputs, training, params).head()
        // X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)

        X = clsToken.array.repeat(0, X.shape[0]).concat(X, 1)
//        X = dropOut.forward(parameterStore, NDList(X.add(posEmbedding.array)), training, params).head()
//        X = blks0.forward(parameterStore, NDList(X), training, params).head()
        X = blks0.forward(parameterStore, NDList(X.add(posEmbedding.array)), training, params).head()
        return head.forward(parameterStore, NDList(X.get(NDIndex(":, 0"))), training, params)
    }

    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape
    ) {
        clsToken.initialize(manager, dataType)
        posEmbedding.initialize(manager, dataType)
        patchEmbedding.initialize(
            manager,
            dataType,
            Shape(inputShapes[0][0], inputShapes[0][1], imgSize.toLong(), imgSize.toLong())
        )
        blks0.initialize(manager, dataType, Shape(inputShapes[0][0], numSteps.toLong(), numHiddens.toLong()))
        head.initialize(manager, dataType, Shape(inputShapes[0][0], numHiddens.toLong()))
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        return arrayOf(Shape(inputShapes[0][0], numClasses.toLong()))
    }
}
