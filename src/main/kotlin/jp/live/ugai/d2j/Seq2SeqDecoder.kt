package jp.live.ugai.d2j

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.Vocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
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

class Seq2SeqDecoder(vocabSize: Int, embedSize: Int, numHiddens: Int, numLayers: Int, dropout: Float) : Decoder() {
    private val embedding: TrainableWordEmbedding
    private val rnn: GRU
    private val dense: Linear

    /* The RNN decoder for sequence to sequence learning. */
    init {
        val list: List<String> = (0 until vocabSize).map { it.toString() }
        val vocab: Vocabulary = DefaultVocabulary(list)
        embedding = TrainableWordEmbedding.builder()
            .optNumEmbeddings(vocabSize)
            .setEmbeddingSize(embedSize)
            .setVocabulary(vocab)
            .build()
        addChildBlock("embedding", embedding)
        rnn = GRU.builder()
            .setNumLayers(numLayers)
            .setStateSize(numHiddens)
            .optReturnState(true)
            .optBatchFirst(false)
            .optDropRate(dropout)
            .build()
        addChildBlock("rnn", rnn)
        dense = Linear.builder().setUnits(vocabSize.toLong()).build()
        addChildBlock("dense", dense)
    }

    /** {@inheritDoc}  */
    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        embedding.initialize(manager, dataType, inputShapes[0])
        manager.newSubManager().use { sub ->
            var shape: Shape = embedding.getOutputShapes(arrayOf(inputShapes[0]))[0]
            val nd = sub.zeros(shape, dataType).swapAxes(0, 1)
            val state = sub.zeros(inputShapes[1], dataType)
            var context = state[NDIndex(-1)]
            context = context.broadcast(
                Shape(
                    nd.shape.head(),
                    context.shape.head(),
                    context.shape[1]
                )
            )
            // Broadcast `context` so it has the same `numSteps` as `X`
            val xAndContext = NDArrays.concat(NDList(nd, context), 2)
            rnn.initialize(manager, dataType, xAndContext.shape)
            shape = rnn.getOutputShapes(arrayOf(xAndContext.shape))[0]
            dense.initialize(manager, dataType, shape)
        }
    }

    override fun initState(encOutputs: NDList): NDList {
        return NDList(encOutputs[1])
    }

    override fun forwardInternal(
        parameterStore: ParameterStore?,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var X = inputs.head()
        var state = inputs[1]
        X = embedding
            .forward(parameterStore, NDList(X), training, params)
            .head()
            .swapAxes(0, 1)
        var context = state[NDIndex(-1)]
        // Broadcast `context` so it has the same `numSteps` as `X`
        context = context.broadcast(
            Shape(
                X.shape.head(),
                context.shape.head(),
                context.shape[1]
            )
        )
        val xAndContext = NDArrays.concat(NDList(X, context), 2)
        val rnnOutput = rnn.forward(parameterStore, NDList(xAndContext, state), training)
        var output = rnnOutput.head()
        state = rnnOutput[1]
        output = dense.forward(parameterStore, NDList(output), training)
            .head()
            .swapAxes(0, 1)
        return NDList(output, state)
    }
}
