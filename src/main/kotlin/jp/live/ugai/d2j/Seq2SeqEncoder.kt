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
    private val numHiddens: Int,
    private val numLayers: Int,
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
     * Initializes the embedding layer and GRU with appropriate input shapes.
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
        // Embedding expects integer indices (int32/int64), not floats.
        input = input.toType(DataType.INT64, false)
        // The output `input` shape: (`batchSize`, `numSteps`, `embedSize`)
        input = embedding.forward(ps, NDList(input), training, params).head()
        // In RNN models, the first axis corresponds to time steps
        input = input.swapAxes(0, 1)
        return rnn.forward(ps, NDList(input), training)
    }

    /**
     * Returns encoder output and state shapes for decoder initialization.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        require(inputShapes.isNotEmpty()) { "Seq2SeqEncoder expects at least one input shape." }
        val input = inputShapes[0]
        val batch = input[0]
        val steps = if (input.dimension() >= 2) input[1] else 1
        val outputs = Shape(steps, batch, numHiddens.toLong())
        val state = Shape(numLayers.toLong(), batch, numHiddens.toLong())
        return arrayOf(outputs, state)
    }
}
