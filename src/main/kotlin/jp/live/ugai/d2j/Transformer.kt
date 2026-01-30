package jp.live.ugai.d2j

import ai.djl.Device
import ai.djl.Model
import ai.djl.engine.Engine
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
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.BatchNorm
import ai.djl.nn.norm.Dropout
import ai.djl.nn.norm.LayerNorm
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import jp.live.ugai.d2j.attention.AttentionDecoder
import jp.live.ugai.d2j.attention.MultiHeadAttention
import jp.live.ugai.d2j.lstm.Encoder
import jp.live.ugai.d2j.lstm.EncoderDecoder
import jp.live.ugai.d2j.timemachine.Vocab
import jp.live.ugai.d2j.util.Accumulator
import jp.live.ugai.d2j.util.NMT
import jp.live.ugai.d2j.util.StopWatch
import jp.live.ugai.d2j.util.TrainingChapter9
import java.util.Locale

/**
 * Executes main.
 */
fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    val manager = NDManager.newBaseManager()
    val ps = ParameterStore(manager, false)

//    net.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
    val ffn0 = positionWiseFFN(4, 8)
    ffn0.initialize(manager, DataType.FLOAT32, Shape(2, 3, 4))
    println(ffn0.forward(ps, NDList(manager.ones(Shape(2, 3, 4))), false)[0][0])

    val ln0 = LayerNorm.builder().build()
    ln0.initialize(manager, DataType.FLOAT32, Shape(2, 2))
    val bn0 = BatchNorm.builder().build()
    bn0.initialize(manager, DataType.FLOAT32, Shape(2, 2))
    val inputMatrix = manager.create(floatArrayOf(1f, 2f, 2f, 3f)).reshape(Shape(2, 2))
    print("LayerNorm: ")
    println(ln0.forward(ps, NDList(inputMatrix), false)[0])
    print("BatchNorm: ")
    println(bn0.forward(ps, NDList(inputMatrix), false)[0])

    val addNorm = AddNorm(0.5f)
    addNorm.initialize(manager, DataType.FLOAT32, Shape(2, 3, 4))
    println(
        addNorm
            .forward(
                ps,
                NDList(manager.ones(Shape(2, 3, 4)), manager.ones(Shape(2, 3, 4))),
                false,
            )[0]
            .shapeEquals(manager.ones(Shape(2, 3, 4))),
    )

    val encoderInput = manager.ones(Shape(2, 100, 24))
    val validLens = manager.create(floatArrayOf(3f, 2f))
    val encoderBlock = TransformerEncoderBlock(24, 48, 8, 0.5f)
    encoderBlock.initialize(manager, DataType.FLOAT32, encoderInput.shape, validLens.shape)
    println(encoderBlock.forward(ps, NDList(encoderInput, validLens), false))

    val encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5f, manager)
    encoder.initialize(manager, DataType.FLOAT32, Shape(2, 100), validLens.shape)
    println(encoder.forward(ps, NDList(manager.ones(Shape(2, 100)), validLens), false))

    val decoderBlk = TransformerDecoderBlock(24, 48, 8, 0.5f, 0)
    val decoderInput = manager.ones(Shape(2, 100, 24))
    val input = NDList(decoderInput, validLens)

    decoderBlk.initialize(manager, DataType.FLOAT32, input.shapes[0], input.shapes[1])
    val state = encoderBlock.forward(ps, NDList(decoderInput, validLens, validLens), false)
    println(decoderBlk.forward(ps, NDList(decoderInput, state.head(), validLens, null), false))

    run {}
}

/**
 * Executes positionWiseFFN.
 */
fun positionWiseFFN(
    ffnNumHiddens: Long,
    ffnNumOutputs: Long,
): AbstractBlock {
    val net = SequentialBlock()
    net.add(Linear.builder().setUnits(ffnNumHiddens).build())
    net.add(Activation::relu)
    net.add(Linear.builder().setUnits(ffnNumOutputs).build())
    return net
}

/**
 * Represents AddNorm.
 */
class AddNorm(
    rate: Float,
) : AbstractBlock() {
    /**
     * The dropout.
     */
    val dropout = Dropout.builder().optRate(rate).build()

    /**
     * The ln.
     */
    val ln = LayerNorm.builder().build()

    init {
        addChildBlock("dropout", dropout)
        addChildBlock("layerNorm", ln)
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
        val x = inputs[0]
        val y = inputs[1]
        val dropoutResult = dropout.forward(ps, NDList(y), training, params).singletonOrThrow()
        val result = ln.forward(ps, NDList(dropoutResult.add(x)), training, params)
        return result
    }

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> = inputShapes

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        require(inputShapes.size == 1) {
            "AddNorm expects a single input shape for initialization (both inputs share the same shape), " +
                "got ${inputShapes.size}."
        }
        dropout.initialize(manager, dataType, inputShapes[0])
        ln.initialize(manager, dataType, inputShapes[0])
    }
}

/**
 * Represents TransformerEncoderBlock.
 */
class TransformerEncoderBlock(
    numHiddens: Int,
    ffnNumHiddens: Long,
    numHeads: Int,
    dropout: Float,
    useBias: Boolean = false,
) : AbstractBlock() {
    /**
     * The attention.
     */
    val attention = MultiHeadAttention(numHiddens, numHeads, dropout, useBias)

    /**
     * The addnorm1.
     */
    val addnorm1 = AddNorm(dropout)

    /**
     * The ffn.
     */
    val ffn = positionWiseFFN(ffnNumHiddens, numHiddens.toLong())

    /**
     * The addnorm2.
     */
    val addnorm2 = AddNorm(dropout)

    init {
        addChildBlock("attention", attention)
        addChildBlock("addnorm1", addnorm1)
        addChildBlock("ffn", ffn)
        addChildBlock("addnorm2", addnorm2)
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
        val x = inputs[0]
        val validLens = inputs[1]
        val y =
            addnorm1.forward(
                ps,
                NDList(x, attention.forward(ps, NDList(x, x, x, validLens), training, params).singletonOrThrow()),
                training,
                params,
            )
        val ret =
            addnorm2.forward(
                ps,
                NDList(y.singletonOrThrow(), ffn.forward(ps, y, training, params).singletonOrThrow()),
                training,
                params,
            )
        return ret
    }

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> = arrayOf(inputShapes[0])

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        val shapes = arrayOf(inputShapes[0], inputShapes[0], inputShapes[0], inputShapes[1])
        attention.initialize(manager, dataType, shapes[0], shapes[1], shapes[2], shapes[3])
        addnorm1.initialize(manager, dataType, inputShapes[0])
        ffn.initialize(manager, dataType, inputShapes[0])
        addnorm2.initialize(manager, dataType, inputShapes[0])
    }
}

/**
 * Represents TransformerEncoder.
 * @property numHiddens The numHiddens.
 */
class TransformerEncoder(
    vocabSize: Int,
    /**
     * The numHiddens.
     */
    val numHiddens: Int,
    ffnNumHiddens: Long,
    numHeads: Long,
    numBlks: Int,
    dropout: Float,
    manager: NDManager,
    useBias: Boolean = false,
) : Encoder() {
    private val embedding: TrainableWordEmbedding

    /**
     * The posEncoding.
     */
    val posEncoding = PositionalEncoding(numHiddens, dropout, 1000, manager)

    /**
     * The blks.
     */
    val blks = mutableListOf<TransformerEncoderBlock>()

    /**
     * The attentionWeights.
     */
    val attentionWeights = Array<NDArray?>(numBlks) { null }

    // The RNN encoder for sequence to sequence learning.
    init {
        val list: List<String> = (0 until vocabSize).map { it.toString() }
        val vocab: Vocabulary = DefaultVocabulary(list)
        // Embedding layer
        embedding =
            TrainableWordEmbedding
                .builder()
                .optNumEmbeddings(vocabSize)
                .setEmbeddingSize(numHiddens)
                .setVocabulary(vocab)
                .build()
        addChildBlock("embedding", embedding)
        addChildBlock("posEncoding", posEncoding)
        repeat(numBlks) {
            val blk = TransformerEncoderBlock(numHiddens, ffnNumHiddens, numHeads.toInt(), dropout, useBias)
            blks.add(blk)
            addChildBlock("block_$it", blk)
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
        var tokenIds = inputs[0]
        val validLens = inputs[1]
        val emb =
            embedding
                .forward(ps, NDList(tokenIds), training, params)
                .singletonOrThrow()
                .mul(Math.sqrt(numHiddens.toDouble()))
        tokenIds = posEncoding.forward(ps, NDList(emb), training, params).singletonOrThrow()
        for (i in 0 until blks.size) {
            tokenIds = blks[i].forward(ps, NDList(tokenIds, validLens), training, params).singletonOrThrow()
            attentionWeights[i] = blks[i].attention.attention.attentionWeights
        }
        return NDList(tokenIds, validLens)
    }

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        require(inputShapes.isNotEmpty()) {
            "TransformerEncoder expects at least one input shape."
        }
        embedding.initialize(manager, dataType, inputShapes[0])
        val modelShape =
            if (inputShapes[0].dimension() == 3) {
                inputShapes[0]
            } else {
                inputShapes[0].add(numHiddens.toLong())
            }
        posEncoding.initialize(manager, dataType, modelShape)
        for (blk in blks) {
            blk.initialize(manager, dataType, modelShape, inputShapes[1])
        }
    }
}

/**
 * Represents TransformerDecoderBlock.
 * @property numHiddens The numHiddens.
 * @property blockIndex The blockIndex.
 */
class TransformerDecoderBlock(
    /**
     * The numHiddens.
     */
    val numHiddens: Int,
    ffnNumHiddens: Long,
    numHeads: Long,
    dropout: Float,
    /**
     * The blockIndex.
     */
    val blockIndex: Int,
) : AbstractBlock() {
    /**
     * The attention1.
     */
    val attention1 = MultiHeadAttention(numHiddens, numHeads.toInt(), dropout, false)

    /**
     * The addnorm1.
     */
    val addnorm1 = AddNorm(dropout)

    /**
     * The attention2.
     */
    val attention2 = MultiHeadAttention(numHiddens, numHeads.toInt(), dropout, false)

    /**
     * The addnorm2.
     */
    val addnorm2 = AddNorm(dropout)

    /**
     * The ffn.
     */
    val ffn = positionWiseFFN(ffnNumHiddens, numHiddens.toLong())

    /**
     * The addnorm3.
     */
    val addnorm3 = AddNorm(dropout)

    init {
        addChildBlock("attention1", attention1)
        addChildBlock("addnorm1", addnorm1)
        addChildBlock("attention2", attention2)
        addChildBlock("addnorm2", addnorm2)
        addChildBlock("ffn", ffn)
        addChildBlock("addnorm3", addnorm3)
    }

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        val decoderShape = inputShapes[0]
        val encoderShape =
            if (inputShapes.size > 1 && inputShapes[1].dimension() == 3) {
                inputShapes[1]
            } else {
                decoderShape
            }
        val encValidLensShape =
            if (inputShapes.size > 2) {
                inputShapes[2]
            } else if (inputShapes.size > 1 && inputShapes[1].dimension() != 3) {
                inputShapes[1]
            } else {
                null
            }

        attention1.initialize(manager, dataType, decoderShape, decoderShape, decoderShape)
        addnorm1.initialize(manager, dataType, decoderShape)
        if (encValidLensShape == null) {
            attention2.initialize(manager, dataType, decoderShape, encoderShape, encoderShape)
        } else {
            attention2.initialize(manager, dataType, decoderShape, encoderShape, encoderShape, encValidLensShape)
        }
        addnorm2.initialize(manager, dataType, decoderShape)
        ffn.initialize(manager, dataType, decoderShape)
        addnorm3.initialize(manager, dataType, decoderShape)
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
        val input0 = inputs[0]
        val encOutputs = inputs[1]
        val envValidLens = if (inputs.size > 2) inputs[2] else null
//        # During training, all the tokens of any output sequence are processed
//        # at the same time, so state[2][self.i] is None as initialized. When
//        # decoding any output sequence token by token during prediction,
//        # state[2][self.i] contains representations of the decoded output at
//        # the i-th block up to the current time step

        // TODO FIX IT
        var keyValues = input0
        if (inputs.size >= 4 && inputs[3] != null) {
            keyValues = NDArrays.concat(NDList(inputs[3], keyValues), 1)
        }

        var decValidLens: NDArray?
        if (training) {
            val batchSize = input0.shape[0]
            val numSteps = input0.shape[1]
            //  Shape of dec_valid_lens: (batch_size, num_steps), where every
            //  row is [1, 2, ..., num_steps]
            decValidLens =
                input0
                    .manager
                    .arange(1f, (numSteps + 1).toFloat())
                    .reshape(1, numSteps)
                    .repeat(0, batchSize)
        } else {
            decValidLens = null
        }
//        # Self-attention
        val selfAttnInputs =
            if (decValidLens == null) {
                NDList(input0, keyValues, keyValues)
            } else {
                NDList(input0, keyValues, keyValues, decValidLens)
            }
        val selfAttnOutput = attention1.forward(ps, selfAttnInputs, training)
        val selfAttnNorm = addnorm1.forward(ps, NDList(input0, selfAttnOutput.head()), training)
//        # Encoder-decoder attention. Shape of enc_outputs:
//        # (batch_size, num_steps, num_hiddens)
        val crossAttnInputs =
            if (envValidLens == null) {
                NDList(selfAttnNorm.head(), encOutputs, encOutputs)
            } else {
                NDList(selfAttnNorm.head(), encOutputs, encOutputs, envValidLens)
            }
        val crossAttnOutput = attention2.forward(ps, crossAttnInputs, training)
        val crossAttnNorm = addnorm2.forward(ps, NDList(selfAttnNorm.head(), crossAttnOutput.head()), training)
        return NDList(
            addnorm3
                .forward(
                    ps,
                    NDList(crossAttnNorm.head(), ffn.forward(ps, NDList(crossAttnNorm), training).head()),
                    training,
                ).head(),
            encOutputs,
            envValidLens,
            keyValues,
        )
    }

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> =
        if (inputShapes.size >= 3) {
            arrayOf(inputShapes[0], inputShapes[1], inputShapes[2], inputShapes[0])
        } else {
            arrayOf(inputShapes[0])
        }
}

/**
 * Represents TransformerDecoder.
 * @property numHiddens The numHiddens.
 * @property numBlks The numBlks.
 */
class TransformerDecoder(
    vocabSize: Int,
    /**
     * The numHiddens.
     */
    val numHiddens: Int,
    ffnNumHiddens: Int,
    numHeads: Int,
    /**
     * The numBlks.
     */
    val numBlks: Int,
    dropout: Float,
    manager: NDManager,
) : AttentionDecoder() {
    /**
     * The list.
     */
    val list: List<String> = (0 until vocabSize).map { it.toString() }

    /**
     * The vocab.
     */
    val vocab: Vocabulary = DefaultVocabulary(list)

    /**
     * The embedding.
     */
    val embedding =
        TrainableWordEmbedding
            .builder()
            .optNumEmbeddings(vocabSize)
            .setEmbeddingSize(numHiddens)
            .setVocabulary(vocab)
            .build()

    /**
     * The posEncoding.
     */
    val posEncoding = PositionalEncoding(numHiddens, dropout, 1000, manager)

    /**
     * The blks.
     */
    val blks = mutableListOf<TransformerDecoderBlock>()

    //            val attentionWeights = Array<NDArray?>(numBlks) { null }

    /**
     * The linear.
     */
    val linear = Linear.builder().setUnits(vocabSize.toLong()).build()

    /**
     * The attentionWeightsArr2.
     */
    var attentionWeightsArr2: MutableList<NDArray?>? = null

    /**
     * The attentionWeightsArr1.
     */
    var attentionWeightsArr1: MutableList<NDArray?>? = null

    init {
        addChildBlock("embedding", embedding)
        addChildBlock("posEncoding", posEncoding)
        repeat(numBlks) {
            /**
             * The blk.
             */
            val blk =
                TransformerDecoderBlock(
                    numHiddens,
                    ffnNumHiddens.toLong(),
                    numHeads.toLong(),
                    dropout,
                    it,
                )
            blks.add(blk)
            addChildBlock("block_$it", blk)
        }
        addChildBlock("linear", linear)
    }

    /**
     * Executes initState.
     */
    override fun initState(encOutputs: NDList): NDList {
        val (encOutputsValue, encValidLens) = encOutputs
        return NDList(encOutputsValue, encValidLens, null)
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
        val inputTokens = inputs[0]
        var state = inputs.subNDList(1)
        val pos =
            posEncoding.forward(
                ps,
                NDList(
                    embedding
                        .forward(ps, NDList(inputTokens), training, params)
                        .head()
                        .mul(Math.sqrt(numHiddens.toDouble())),
                ),
                training,
                params,
            )
        attentionWeightsArr1 = mutableListOf()
        attentionWeightsArr2 = mutableListOf()
        var outX = pos.head()
        for (i in 0 until blks.size) {
            val blk = blks[i].forward(ps, NDList(outX).addAll(state), training, params)
            outX = blk.head()
            state = blk.subNDList(1)
            attentionWeightsArr1!!.add(blks[i].attention1.attention.attentionWeights)
            attentionWeightsArr2!!.add(blks[i].attention2.attention.attentionWeights)
        }
        val ret = linear.forward(ps, NDList(outX), training, params)
        return NDList(ret.head()).addAll(state)
    }

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        val tokenShape =
            if (inputShapes[0].dimension() == 3) {
                Shape(inputShapes[0][0], inputShapes[0][1])
            } else {
                inputShapes[0]
            }
        val modelShape =
            if (inputShapes[0].dimension() == 3) {
                inputShapes[0]
            } else {
                inputShapes[0].add(numHiddens.toLong())
            }
        embedding.initialize(manager, dataType, tokenShape)
        posEncoding.initialize(manager, dataType, modelShape)
        for (blk in blks) {
            val encoderShape =
                if (inputShapes.size > 2) {
                    inputShapes[1]
                } else if (inputShapes.size > 1 && inputShapes[1].dimension() == 3) {
                    inputShapes[1]
                } else {
                    modelShape
                }
            val validLensShape =
                if (inputShapes.size > 2) {
                    inputShapes[2]
                } else if (inputShapes.size > 1 && inputShapes[1].dimension() != 3) {
                    inputShapes[1]
                } else {
                    null
                }
            if (validLensShape == null) {
                blk.initialize(manager, dataType, modelShape, encoderShape)
            } else {
                blk.initialize(manager, dataType, modelShape, encoderShape, validLensShape)
            }
        }
        linear.initialize(manager, dataType, modelShape)
    }
}

/**
 * Executes train.
 */
fun train() {
//        num_hiddens, num_blks, dropout = 256, 2, 0.2
//        ffn_num_hiddens, num_heads = 64, 4
    val manager = NDManager.newBaseManager()

    val numHiddens = 256
    val numBlks = 2
    val ffnNumHiddens = 64
    val numHeads = 4
    val batchSize = 2
    val numEpochs = 30
    val numSteps = 35
    println(numSteps)

    val dropout = 0.2f
    val lr = 0.001f
    val device = manager.device

    val dataNMT = NMT.loadDataNMT(batchSize, numSteps, 600)
    val dataset: ArrayDataset = dataNMT.first
    val srcVocab: Vocab = dataNMT.second.first
    val tgtVocab: Vocab = dataNMT.second.second

    val encoder =
        TransformerEncoder(
            srcVocab.length(),
            numHiddens,
            ffnNumHiddens.toLong(),
            numHeads.toLong(),
            numBlks,
            dropout,
            manager,
        )
    encoder.initialize(manager, DataType.FLOAT32, Shape(2, 35), Shape(2))

    val decoder = TransformerDecoder(tgtVocab.length(), numHiddens, ffnNumHiddens, numHeads, numBlks, dropout, manager)
    decoder.initialize(manager, DataType.FLOAT32, Shape(2, 35, 256), Shape(2, 35, 256), Shape(2))

    val net = EncoderDecoder(encoder, decoder)
    trainSeq2Seq(net, dataset, lr, numEpochs, tgtVocab, device)

    val engs = arrayOf("go .", "i lost .", "he's calm .", "i'm home .")
    val fras = arrayOf("va !", "j'ai perdu .", "il est calme .", "je suis chez moi .")
    for (i in engs.indices) {
        val pair = predictSeq2Seq(net, engs[i], srcVocab, tgtVocab, numSteps, false)
        val translation: String = pair.first
        println("%s => %s, bleu %.3f".format(engs[i], translation, bleu(translation, fras[i], 2)))
    }
}

/**
 * Executes trainSeq2Seq.
 */
fun trainSeq2Seq(
    net: EncoderDecoder,
    dataset: ArrayDataset,
    lr: Float,
    numEpochs: Int,
    tgtVocab: Vocab,
    device: Device,
) {
    val manager = NDManager.newBaseManager()
    val loss: Loss = MaskedSoftmaxCELoss()
    val lrt: Tracker = Tracker.fixed(lr)
    val adam: Optimizer = Optimizer.adam().optLearningRateTracker(lrt).build()
    val config: DefaultTrainingConfig =
        DefaultTrainingConfig(loss)
            .optOptimizer(adam) // Optimizer (loss function)
            .optInitializer(XavierInitializer(), "")
    val model: Model = Model.newInstance("")
    model.block = net
    val trainer: Trainer = model.newTrainer(config)
//    val animator = Animator()
    var watch: StopWatch
    var metric: Accumulator
    var lossValue = 0.0
    var speed = 0.0
    for (epoch in 1..numEpochs) {
        watch = StopWatch()
        metric = Accumulator(2) // Sum of training loss, no. of tokens
        // Iterate over dataset
        for (batch in dataset.getData(manager)) {
            val sourceBatch: NDArray = batch.data.get(0)
            val sourceValidLen: NDArray = batch.data.get(1)
            val targetBatch: NDArray = batch.labels.get(0)
            val targetValidLen: NDArray = batch.labels.get(1)
            val bos: NDArray =
                manager
                    .full(Shape(targetBatch.shape[0]), tgtVocab.getIdx("<bos>"))
                    .reshape(-1, 1)
            val decInput: NDArray =
                NDArrays.concat(
                    NDList(bos, targetBatch.get(NDIndex(":, :-1"))),
                    1,
                ) // Teacher forcing
            Engine.getInstance().newGradientCollector().use { gc ->
                val yHat: NDArray =
                    net
                        .forward(
                            ParameterStore(manager, false),
                            NDList(sourceBatch, decInput, sourceValidLen),
                            true,
                        ).get(0)
                val l = loss.evaluate(NDList(targetBatch, targetValidLen), NDList(yHat))
                gc.backward(l)
                metric.add(floatArrayOf(l.sum().getFloat(), targetValidLen.sum().getLong().toFloat()))
            }
            TrainingChapter9.gradClipping(net, 1, manager)
            // Update parameters
            trainer.step()
        }
        lossValue = metric.get(0).toDouble() / metric.get(1)
        speed = metric.get(1) / watch.stop()
//                if ((epoch + 1) % 10 == 0) {
//            animator.add(epoch + 1, lossValue.toFloat(), "loss")
//            animator.show()
        println("${epoch + 1} : $lossValue")
//                }
    }
    println("loss: %.3f, %.1f tokens/sec on %s%n".format(lossValue, speed, device.toString()))
}

/**
 * Executes predictSeq2Seq.
 */
fun predictSeq2Seq(
    net: EncoderDecoder,
    srcSentence: String,
    srcVocab: Vocab,
    tgtVocab: Vocab,
    numSteps: Int,
    saveAttentionWeights: Boolean,
): Pair<String, List<NDArray?>> {
    val manager = NDManager.newBaseManager()
    val srcTokens =
        srcVocab.getIdxs(srcSentence.lowercase(Locale.getDefault()).split(" ")) +
            listOf(srcVocab.getIdx("<eos>"))
    val encValidLen = manager.create(srcTokens.size).reshape(1)
    val truncateSrcTokens = NMT.truncatePad(srcTokens, numSteps, srcVocab.getIdx("<pad>"))
    // Add the batch axis
    val encX = manager.create(truncateSrcTokens.toIntArray()).expandDims(0)
    val encOutputs = net.encoder.forward(ParameterStore(manager, false), NDList(encX, encValidLen), false)
    var decState = net.decoder.initState(encOutputs)
    // Add the batch axis
    var decX = manager.create(floatArrayOf(tgtVocab.getIdx("<bos>").toFloat())).reshape(1, 1)
    val outputSeq: MutableList<Int> = mutableListOf()
    val attentionWeightSeq: MutableList<NDArray?> = mutableListOf()
    for (i in 0 until numSteps) {
        val output =
            net.decoder.forward(
                ParameterStore(manager, false),
                NDList(decX).addAll(decState),
                false,
            )
        val decoderOutput = output[0]
        decState = output.subNDList(1)
        // We use the token with the highest prediction likelihood as the input
        // of the decoder at the next time step
        decX = decoderOutput.argMax(2)
        val pred = decX.squeeze(0).getLong().toInt()
        // Save attention weights (to be covered later)
        if (saveAttentionWeights) {
//            attentionWeightSeq.add(net.decoder.attentionWeights)
        }
        // Once the end-of-sequence token is predicted, the generation of the
        // output sequence is complete
        if (pred == tgtVocab.getIdx("<eos>")) {
            break
        }
        outputSeq.add(pred)
    }
    val outputString: String = tgtVocab.toTokens(outputSeq).joinToString(separator = " ")
    return Pair(outputString, attentionWeightSeq.toList())
}

// Compute the BLEU.

/**
 * Executes bleu.
 */
fun bleu(
    predSeq: String,
    labelSeq: String,
    k: Int,
): Double {
    val predTokens = predSeq.split(" ")
    val labelTokens = labelSeq.split(" ")
    val lenPred = predTokens.size
    val lenLabel = labelTokens.size
    var score = Math.exp(Math.min(0.toDouble(), 1.0 - lenLabel / lenPred))
    for (n in 1 until k + 1) {
        var numMatches = 0
        val labelSubs = mutableMapOf<String, Int>()
        for (i in 0 until lenLabel - n + 1) {
            val key = labelTokens.subList(i, i + n).joinToString(separator = " ")
            labelSubs.put(key, labelSubs.getOrDefault(key, 0) + 1)
        }
        for (i in 0 until lenPred - n + 1) {
            // val key =predTokens.subList(i, i + n).joinToString(" ")
            val key = predTokens.subList(i, i + n).joinToString(separator = " ")
            if (labelSubs.getOrDefault(key, 0) > 0) {
                numMatches += 1
                labelSubs.put(key, labelSubs.getOrDefault(key, 0) - 1)
            }
        }
        score *= Math.pow(numMatches.toDouble() / (lenPred - n + 1).toDouble(), Math.pow(0.5, n.toDouble()))
    }
    return score
}
