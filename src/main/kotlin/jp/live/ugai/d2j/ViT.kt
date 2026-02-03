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

/**
 * Represents ViT.
 * @property imgSize The imgSize.
 * @property numHiddens The numHiddens.
 * @property numClasses The numClasses.
 */
class ViT(
    /**
     * The imgSize.
     */
    val imgSize: Int,
    patchSize: Int,
    /**
     * The numHiddens.
     */
    val numHiddens: Int,
    mlpNumHiddens: Int,
    numHeads: Int,
    numBlks: Int,
    embDropout: Float,
    blkDropout: Float,
    useBias: Boolean = false,
    /**
     * The numClasses.
     */
    val numClasses: Int = 10,
) : AbstractBlock() {
    /**
     * The patchEmbedding.
     */
    val patchEmbedding = PatchEmbedding(imgSize, patchSize, numHiddens)

    /**
     * The clsToken.
     */
    val clsToken =
        Parameter
            .builder()
            .optRequiresGrad(true)
            .setType(Parameter.Type.BIAS)
            .optShape(Shape(1, 1, numHiddens.toLong()))
            .build()

    /**
     * The numSteps.
     */
    val numSteps: Int = patchEmbedding.numPatches + 1

    /**
     * The posEmbedding.
     */
    val posEmbedding =
        Parameter
            .builder()
            .optRequiresGrad(true)
            .optShape(Shape(1, numSteps.toLong(), numHiddens.toLong()))
            .setType(Parameter.Type.BIAS)
            .build()

    /**
     * The blks0.
     */
    val blks0 =
        SequentialBlock()
            .add(Dropout.builder().optRate(embDropout).build())

    /**
     * The head.
     */
    val head =
        SequentialBlock()
            .add(LayerNorm.builder().build())
            .add(Linear.builder().setUnits(numClasses.toLong()).build())

    init {
        addParameter(clsToken)
        addParameter(posEmbedding)
        clsToken.setInitializer(ConstantInitializer(0f))
        addChildBlock("patchEmbedding", patchEmbedding)
        addChildBlock("blocks", blks0)
        addChildBlock("head", head)
        repeat(numBlks) {
            blks0.add(ViTBlock(numHiddens, numHiddens, mlpNumHiddens, numHeads, blkDropout, useBias))
        }
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
        logGpu("ViT pre-patch")
        var embeddings = patchEmbedding.forward(parameterStore, inputs, training, params).head()
        logGpu("ViT post-patch")
        val device = inputs.head().device
        val clsTokenArray = parameterStore.getValue(clsToken, device, training)
        val posEmbeddingArray = parameterStore.getValue(posEmbedding, device, training)
        // embeddings = torch.cat((self.cls_token.expand(embeddings.shape[0], -1, -1), embeddings), 1)

        embeddings = clsTokenArray.repeat(0, embeddings.shape[0]).concat(embeddings, 1)
        logGpu("ViT post-cls")
        embeddings =
            blks0
                .forward(
                    parameterStore,
                    NDList(embeddings.add(posEmbeddingArray)),
                    training,
                    params,
                ).head()
        logGpu("ViT post-blocks")
        val out = head.forward(parameterStore, NDList(embeddings.get(NDIndex(":, 0"))), training, params)
        logGpu("ViT post-head")
        return out
    }

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        clsToken.initialize(manager, dataType)
        posEmbedding.initialize(manager, dataType)
        patchEmbedding.initialize(
            manager,
            dataType,
            Shape(inputShapes[0][0], inputShapes[0][1], inputShapes[0][2], inputShapes[0][3]),
        )
        blks0.initialize(
            manager,
            dataType,
            Shape(inputShapes[0][0], numSteps.toLong(), numHiddens.toLong()),
        )
        head.initialize(manager, dataType, Shape(inputShapes[0][0], numHiddens.toLong()))
    }

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> = arrayOf(Shape(inputShapes[0][0], numClasses.toLong()))
}
