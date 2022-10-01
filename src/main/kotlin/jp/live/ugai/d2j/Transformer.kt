package jp.live.ugai.d2j

import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.Vocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.BatchNorm
import ai.djl.nn.norm.Dropout
import ai.djl.nn.norm.LayerNorm
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import jp.live.ugai.d2j.attention.MultiHeadAttention
import jp.live.ugai.d2j.lstm.Encoder

fun main() {
    val manager = NDManager.newBaseManager()
    val ps = ParameterStore(manager, false)

    fun positionWiseFFN(ffn_num_hiddens: Long, ffn_num_outputs: Long): AbstractBlock {
        val net = SequentialBlock()
        net.add(Linear.builder().setUnits(ffn_num_hiddens).build())
        net.add(Activation::relu)
        net.add(Linear.builder().setUnits(ffn_num_outputs).build())
        return net
    }
//    net.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
    val ffn0 = positionWiseFFN(4, 8)
    ffn0.initialize(manager, DataType.FLOAT32, Shape(2, 3, 4))
    println(ffn0.forward(ps, NDList(manager.ones(Shape(2, 3, 4))), false)[0][0])

    val ln0 = LayerNorm.builder().build()
    ln0.initialize(manager, DataType.FLOAT32, Shape(2, 2))
    val bn0 = BatchNorm.builder().build()
    bn0.initialize(manager, DataType.FLOAT32, Shape(2, 2))
    val X0 = manager.create(floatArrayOf(1f, 2f, 2f, 3f)).reshape(Shape(2, 2))
    print("LayerNorm: ")
    println(ln0.forward(ps, NDList(X0), false)[0])
    print("BatchNorm: ")
    println(bn0.forward(ps, NDList(X0), false)[0])

    class AddNorm(rate: Float) : AbstractBlock() {
        val dropout = Dropout.builder().optRate(rate).build()
        val ln = LayerNorm.builder().build()

        init {
            addChildBlock("dropout", dropout)
            addChildBlock("layerNorm", ln)
        }

        override fun forwardInternal(
            ps: ParameterStore,
            inputs: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            val x = inputs[0]
            val y = inputs[1]
            val dropoutResult = dropout.forward(ps, NDList(y), training, params).singletonOrThrow()
            val result = ln.forward(ps, NDList(dropoutResult.add(x)), training, params)
            return result
        }

        override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
            return arrayOf<Shape>()
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
            val sub = manager.newSubManager()
            dropout.initialize(manager, dataType, *inputShapes)
            ln.initialize(manager, dataType, *inputShapes)
            sub.close()
        }
    }

    val addNorm = AddNorm(0.5f)
    addNorm.initialize(manager, DataType.FLOAT32, Shape(1, 4))
    println(addNorm.forward(ps, NDList(manager.ones(Shape(2, 3, 4)), manager.ones(Shape(2, 3, 4))), false)[0].shapeEquals(manager.ones(Shape(2, 3, 4))))

    class TransformerEncoderBlock(
        numHiddens: Int,
        ffnNumHiddens: Long,
        numHeads: Int,
        dropout: Float,
        useBias: Boolean = false
    ) : AbstractBlock() {
        val attention = MultiHeadAttention(numHiddens, numHeads, dropout, useBias)
        val addnorm1 = AddNorm(dropout)
        val ffn = positionWiseFFN(ffnNumHiddens, numHiddens.toLong())
        val addnorm2 = AddNorm(dropout)
        init {
            addChildBlock("attention", attention)
            addChildBlock("addnorm1", addnorm1)
            addChildBlock("ffn", ffn)
            addChildBlock("addnorm2", addnorm2)
        }

        override fun forwardInternal(
            ps: ParameterStore,
            inputs: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            val x = inputs[0]
            val validLens = inputs[1]
            val y = addnorm1.forward(ps, NDList(x, attention.forward(ps, NDList(x, x, x, validLens), training, params).singletonOrThrow()), training, params)
            val ret = addnorm2.forward(ps, NDList(y.singletonOrThrow(), ffn.forward(ps, y, training, params).singletonOrThrow()), training, params)
            return ret
        }

        override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
            return arrayOf<Shape>()
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
            val shapes = arrayOf(inputShapes[0], inputShapes[0], inputShapes[0], inputShapes[1])
            attention.initialize(manager, dataType, *shapes)
            addnorm1.initialize(manager, dataType, inputShapes[0])
            ffn.initialize(manager, dataType, inputShapes[0])
            addnorm2.initialize(manager, dataType, inputShapes[0])
        }
    }

    val X1 = manager.ones(Shape(2, 100, 24))
    val validLens = manager.create(floatArrayOf(3f, 2f))
    val encoderBlock = TransformerEncoderBlock(24, 48, 8, 0.5f)
    encoderBlock.initialize(manager, DataType.FLOAT32, X1.shape, validLens.shape)
    println(encoderBlock.forward(ps, NDList(X1, validLens), false))

    class TransformerEncoder(
        vocabSize: Int,
        val numHiddens: Int,
        ffnNumHiddens: Long,
        numHeads: Long,
        numBlks: Int,
        dropout: Float,
        useBias: Boolean = false
    ) : Encoder() {

        private val embedding: TrainableWordEmbedding
        val posEncoding = PositionalEncoding(numHiddens, dropout, 1000, manager)
        val blks = mutableListOf<TransformerEncoderBlock>()
        val attentionWeights = Array<NDArray?>(numBlks) { null }

        /* The RNN encoder for sequence to sequence learning. */
        init {
            val list: List<String> = (0 until vocabSize).map { it.toString() }
            val vocab: Vocabulary = DefaultVocabulary(list)
            // Embedding layer
            embedding = TrainableWordEmbedding.builder()
                .optNumEmbeddings(vocabSize)
                .setEmbeddingSize(numHiddens)
                .setVocabulary(vocab)
                .build()
            addChildBlock("embedding", embedding)
            repeat(numBlks) {
                blks.add(TransformerEncoderBlock(numHiddens, ffnNumHiddens, numHeads.toInt(), dropout, useBias))
            }
        }

        override fun forwardInternal(
            ps: ParameterStore,
            inputs: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            var X = inputs[0]
            val validLens = inputs[1]
            val emb = embedding.forward(ps, NDList(X), training, params).singletonOrThrow().mul(Math.sqrt(numHiddens.toDouble()))
            X = posEncoding.forward(ps, NDList(emb), training, params).singletonOrThrow()
            for (i in 0 until blks.size) {
                X = blks[i].forward(ps, NDList(X, validLens), training, params).singletonOrThrow()
                attentionWeights[i] = blks[i].attention.attention.attentionWeights
            }
            return NDList(X)
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
            embedding.initialize(manager, dataType, *inputShapes)
            for (blk in blks) {
                blk.initialize(manager, dataType, inputShapes[0].add(numHiddens.toLong()), inputShapes[1])
            }
        }
    }

    val encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5f)
    encoder.initialize(manager, DataType.FLOAT32, Shape(2, 100), validLens.shape)
    println(encoder.forward(ps, NDList(manager.ones(Shape(2, 100)), validLens), false))

    class TransformerDecoderBlock(
        val numHiddens: Int,
        ffnNumHiddens: Long,
        numHeads: Long,
        dropout: Float,
        _i: Int
    ) : AbstractBlock() {
        val i = _i
        val attention1 = MultiHeadAttention(numHiddens, numHeads.toInt(), dropout, false)
        val addnorm1 = AddNorm(dropout)
        val attention2 = MultiHeadAttention(numHiddens, numHeads.toInt(), dropout, false)
        val addnorm2 = AddNorm(dropout)
        val ffn = positionWiseFFN(ffnNumHiddens, numHiddens.toLong())
        val addnorm3 = AddNorm(dropout)

        init {
            addChildBlock("attention1", attention1)
            addChildBlock("addnorm1", addnorm1)
            addChildBlock("attention2", attention2)
            addChildBlock("addnorm2", addnorm2)
            addChildBlock("ffn", ffn)
            addChildBlock("addnorm3", addnorm3)
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
            val shapes1 = arrayOf(inputShapes[0], inputShapes[0], inputShapes[0], inputShapes[1])
            attention1.initialize(manager, dataType, *shapes1)
            addnorm1.initialize(manager, dataType, inputShapes[0])
            attention2.initialize(manager, dataType, *shapes1)
            addnorm2.initialize(manager, dataType, inputShapes[0])
            ffn.initialize(manager, dataType, inputShapes[0])
            addnorm3.initialize(manager, dataType, inputShapes[0])
        }
        override fun forwardInternal(
            ps: ParameterStore,
            inputs: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            val input0 = inputs[0]
            val encOutputs = inputs[1]
            val envValidLens = inputs[2]
//        # During training, all the tokens of any output sequence are processed
//        # at the same time, so state[2][self.i] is None as initialized. When
//        # decoding any output sequence token by token during prediction,
//        # state[2][self.i] contains representations of the decoded output at
//        # the i-th block up to the current time step
            var keyValues: NDArray? = null
            if (inputs.size < 4 || inputs[3] == null) {
                keyValues = inputs[0]
            } else if (inputs[3]!!.size() < i.toLong()) {
                keyValues = inputs[3].concat(inputs[0])
            } else {
                val keyValue = inputs[3].get(i.toLong()).concat(input0, 1)
                keyValues!!.set(NDIndex(i.toLong()), keyValue)
            }

            var decValidLens: NDArray?
            if (training) {
                val batchSize = input0.shape[0]
                val numSteps = input0.shape[1]
                //  Shape of dec_valid_lens: (batch_size, num_steps), where every
                //  row is [1, 2, ..., num_steps]
                decValidLens = manager.arange(1f, (numSteps + 1).toFloat()).repeat(batchSize).reshape(Shape(batchSize, numSteps))
            } else {
                decValidLens = null
            }
//        # Self-attention
            val X2 = attention1.forward(ps, NDList(input0, keyValues, keyValues, decValidLens), training)
            val Y = addnorm1.forward(ps, NDList(input0, X2.head()), training)
//        # Encoder-decoder attention. Shape of enc_outputs:
//        # (batch_size, num_steps, num_hiddens)
            val Y2 = attention2.forward(ps, NDList(Y.head(), encOutputs, encOutputs, envValidLens), training)
            val Z = addnorm2.forward(ps, NDList(Y.head(), Y2.head()), training)
            return NDList(addnorm3.forward(ps, NDList(Z.head(), ffn.forward(ps, NDList(Z), training).head()), training).head(), encOutputs, envValidLens, keyValues)
        }

        override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
            return arrayOf<Shape>()
        }
    }

    val decoderBlk = TransformerDecoderBlock(24, 48, 8, 0.5f, 0)
    val X = manager.ones(Shape(2, 100, 24))
    val input = NDList(X, validLens)

    decoderBlk.initialize(manager, DataType.FLOAT32, *input.shapes)
    val state = encoderBlock.forward(ps, NDList(X, validLens, validLens), false)
    println(decoderBlk.forward(ps, NDList(X, state.head(), validLens, null), false))
}

class Transformer
