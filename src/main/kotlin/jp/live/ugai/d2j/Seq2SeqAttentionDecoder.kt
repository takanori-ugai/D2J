package jp.live.ugai.d2j

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.Vocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.recurrent.GRU
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import jp.live.ugai.d2j.attention.AdditiveAttention

fun main() {
    val manager = NDManager.newBaseManager()
    val ps = ParameterStore(manager, false)
    val vocabSize = 10
    val embedSize = 8
    val numHiddens = 16
    val numLayers = 2
    val batchSize = 4
    val numSteps = 7
    val encoder = Seq2SeqEncoder(vocabSize, embedSize, numHiddens, numLayers, 0f)
    encoder.initialize(manager, DataType.FLOAT32, Shape(batchSize.toLong(), batchSize.toLong()))
    val decoder = Seq2SeqAttentionDecoder(vocabSize.toLong(), embedSize, numHiddens, numLayers)
    decoder.initialize(
        manager,
        DataType.FLOAT32,
        Shape(batchSize.toLong(), numHiddens.toLong()),
        Shape(batchSize.toLong(), 1, numHiddens.toLong()),
        Shape(1, batchSize.toLong(), numHiddens.toLong()),
        Shape(1, batchSize.toLong(), (numHiddens + embedSize).toLong()),
        Shape(batchSize.toLong(), numSteps.toLong(), numHiddens.toLong())
    )
    val X = manager.zeros(Shape(batchSize.toLong(), numSteps.toLong()))
    val output = encoder.forward(ps, NDList(X), false)
    output.add(manager.create(0))
    val state = decoder.initState(output)
    println("State: $state")
    val ff = decoder.forward(ps, NDList(X).addAll(state), false)
    println(ff)
    println(ff[0].shape) // (batch_size, num_steps, vocab_size) (4, 7, 10)
    println(ff[1].shape) // (batch_size, num_steps, num_hiddens) (4, 7, 16)
    println(ff[2][0].shape) // (batch_size, num_hiddens) (4, 16)
}

abstract class AttentionDecoder : AbstractBlock() {
    var attentionWeights: MutableList<NDArray?>? = null
    abstract fun initState(encOutputs: NDList): NDList
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }
}

class Seq2SeqAttentionDecoder(
    vocabSize: Long,
    embedSize: Int,
    numHiddens: Int,
    numLayers: Int,
    dropout: Float = 0f
) : AttentionDecoder() {
    val attention = AdditiveAttention(numHiddens, dropout)
    val embedding: TrainableWordEmbedding
    val rnn = GRU.builder()
        .setNumLayers(numLayers)
        .setStateSize(numHiddens)
        .optReturnState(true)
        .optBatchFirst(false)
        .optDropRate(dropout)
        .build()
    val linear = Linear.builder().setUnits(vocabSize).build()

    init {
        val list: List<String> = (0 until vocabSize).map { it.toString() }
        val vocab: Vocabulary = DefaultVocabulary(list)
        // Embedding layer
        embedding = TrainableWordEmbedding.builder()
            .optNumEmbeddings(vocabSize.toInt())
            .setEmbeddingSize(embedSize)
            .setVocabulary(vocab)
            .build()
//        addChildBlock("embedding", embedding)
    }

    override fun initState(encOutputs: NDList): NDList {
        val outputs = encOutputs[0]
        val hiddenState = encOutputs[1]
        val encValidLens = encOutputs[2]
        return NDList(outputs.swapAxes(0, 1), hiddenState, encValidLens)
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        embedding.initialize(manager, dataType, inputShapes[0])
        attention.initialize(manager, DataType.FLOAT32, inputShapes[1], inputShapes[2])
        rnn.initialize(manager, DataType.FLOAT32, inputShapes[3])
        linear.initialize(manager, DataType.FLOAT32, inputShapes[4])
    }

    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        attentionWeights = mutableListOf()
        var outputs: NDArray? = null
        val encOutputs = inputs[1]
        var hiddenState: NDArray = inputs[2]
        val encValidLens = inputs[3]
        var input = inputs[0]
//        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
//        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
//        enc_outputs, hidden_state, enc_valid_lens = state
//        # Shape of the output X: (num_steps, batch_size, embed_size)
//        X = self.embedding(X).permute(1, 0, 2)
        // The output `X` shape: (`batchSize`(4), `numSteps`(7), `embedSize`(8))
        val X = embedding.forward(ps, NDList(input), training, params)[0].swapAxes(0, 1)
        for (x in 0 until X.size(0)) {
            val query = hiddenState[-1].expandDims(1)
            val context = attention.forward(ps, NDList(query, encOutputs, encOutputs, encValidLens), training, params)
            val xArray = context[0].concat(X[x].expandDims(1), -1)
            val out = rnn.forward(ps, NDList(xArray.swapAxes(0, 1), hiddenState), training, params)
            hiddenState = out[1]
            outputs = if (outputs == null) out[0] else outputs.concat(out[0])
            attentionWeights!!.add(attention.attentionWeights)
        }
        val ret = linear.forward(ps, NDList(outputs), training)
        return NDList(ret[0].swapAxes(0, 1), encOutputs, hiddenState, encValidLens)
    }
}
