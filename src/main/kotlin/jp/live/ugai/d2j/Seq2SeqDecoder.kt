package jp.live.ugai.d2j

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.Vocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.core.Linear
import ai.djl.nn.recurrent.GRU
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import jp.live.ugai.d2j.lstm.Decoder

/**
 * Represents Seq2SeqDecoder.
 */
class Seq2SeqDecoder(
    vocabSize: Int,
    embedSize: Int,
    numHiddens: Int,
    numLayers: Int,
    dropout: Float,
) : Decoder() {
    private val embedding: TrainableWordEmbedding
    private val rnn: GRU
    private val dense: Linear

    /**
     * The attentionWeights.
     */
    public override var attentionWeights: NDArray? = null

    init {
        val list: List<String> = (0 until vocabSize).map { it.toString() }
        val vocab: Vocabulary = DefaultVocabulary(list)
        embedding =
            TrainableWordEmbedding
                .builder()
                .optNumEmbeddings(vocabSize)
                .setEmbeddingSize(embedSize)
                .setVocabulary(vocab)
                .build()
        addChildBlock("embedding", embedding)
        rnn =
            GRU
                .builder()
                .setNumLayers(numLayers)
                .setStateSize(numHiddens)
                .optReturnState(true)
                .optBatchFirst(false)
                .optDropRate(dropout)
                .build()
        addChildBlock("rnn", rnn)
        dense =
            Linear
                .builder()
                .setUnits(vocabSize.toLong())
                .build()
        addChildBlock("dense", dense)
    }

    /**
     * Initializes embedding, recurrent, and projection layers based on input shapes.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        embedding.initialize(manager, dataType, inputShapes[0])
        manager.newSubManager().use { sub ->
            val shape: Shape = embedding.getOutputShapes(arrayOf(inputShapes[0]))[0]
            val nd = sub.zeros(shape, dataType).swapAxes(0, 1)
            val state = sub.zeros(inputShapes[1], dataType)
            var context = state[NDIndex(-1)]
            context =
                context.broadcast(
                    Shape(
                        nd.shape.head(),
                        context.shape.head(),
                        context.shape[1],
                    ),
                )
            // Broadcast `context` so it has the same `numSteps` as `X`
            val xAndContext = NDArrays.concat(NDList(nd, context), 2)
            rnn.initialize(manager, dataType, xAndContext.shape)
            val rnnOutputShape: Shape = rnn.getOutputShapes(arrayOf(xAndContext.shape))[0]
            dense.initialize(manager, dataType, rnnOutputShape)
        }
    }

    /**
     * Initializes decoder state from encoder outputs.
     */
    override fun initState(encOutputs: NDList): NDList = NDList(encOutputs[1])

    /**
     * Runs the decoder forward pass with attention context concatenated to embeddings.
     */
    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        var input = inputs.head()
        var state = inputs[1]
        // Embedding expects integer indices (int32/int64), not floats.
        input = input.toType(DataType.INT64, false)
        input =
            embedding
                .forward(parameterStore, NDList(input), training, params)
                .head()
                .swapAxes(0, 1)
        var context = state[NDIndex(-1)]
        // Broadcast `context` so it has the same `numSteps` as `input`
        context =
            context.broadcast(
                Shape(
                    input.shape.head(),
                    context.shape.head(),
                    context.shape[1],
                ),
            )
        val xAndContext = NDArrays.concat(NDList(input, context), 2)
        val rnnOutput =
            rnn.forward(parameterStore, NDList(xAndContext, state), training)
        var output = rnnOutput.head()
        state = rnnOutput[1]
        output =
            dense
                .forward(parameterStore, NDList(output), training)
                .head()
                .swapAxes(0, 1)
        return NDList(output, state)
    }
}
