package jp.live.ugai.d2j

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.Vocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.recurrent.GRU
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import jp.live.ugai.d2j.lstm.Encoder

/**
 * The RNN encoder for sequence-to-sequence learning.
 *
 * @property vocabSize The size of the vocabulary.
 * @property embedSize The size of the embedding.
 * @property numHiddens The number of hidden units.
 * @property numLayers The number of layers.
 * @property dropout The dropout rate.
 */
class Seq2SeqEncoder(
    vocabSize: Int,
    embedSize: Int,
    numHiddens: Int,
    numLayers: Int,
    dropout: Float,
) : Encoder() {
    private val embedding: TrainableWordEmbedding
    private val rnn: GRU

    init {
        val list: List<String> = (0 until vocabSize).map { it.toString() }
        val vocab: Vocabulary = DefaultVocabulary(list)
        // Embedding layer
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
    }

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        embedding.initialize(manager, dataType, inputShapes[0])
        val shapes: Array<Shape> = embedding.getOutputShapes(arrayOf(inputShapes[0]))
        manager.newSubManager().use { sub ->
            var nd = sub.zeros(shapes[0], dataType)
            nd = nd.swapAxes(0, 1)
            rnn.initialize(manager, dataType, nd.shape)
        }
    }

    /**
     * Executes forwardInternal.
     */
    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        var input = inputs.head()
        // The output `input` shape: (`batchSize`, `numSteps`, `embedSize`)
        input = embedding.forward(ps, NDList(input), training, params).head()
        // In RNN models, the first axis corresponds to time steps
        input = input.swapAxes(0, 1)
        return rnn.forward(ps, NDList(input), training)
    }
}
