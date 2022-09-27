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
import ai.djl.nn.core.Linear
import ai.djl.nn.recurrent.GRU
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import jp.live.ugai.d2j.attention.AdditiveAttention
import jp.live.ugai.d2j.lstm.Decoder
import jp.live.ugai.d2j.lstm.EncoderDecoder
import jp.live.ugai.d2j.timemachine.Vocab
import jp.live.ugai.d2j.util.Accumulator
import jp.live.ugai.d2j.util.NMT
import jp.live.ugai.d2j.util.StopWatch
import jp.live.ugai.d2j.util.TrainingChapter9
import java.util.*

fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

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
        Shape(batchSize.toLong(), batchSize.toLong(), numHiddens.toLong()),
        Shape(1, batchSize.toLong(), (numHiddens + embedSize).toLong()),
        Shape(4, numHiddens.toLong())
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
    train()
}

fun train() {
    val embedSize = 32
    val numHiddens = 32
    val numLayers = 2
    val batchSize = 64
    val numSteps = 10
    val numEpochs = Integer.getInteger("MAX_EPOCH", 300)

    val dropout = 0.2f
    val lr = 0.01f
    val device = manager.device

    val dataNMT = NMT.loadDataNMT(batchSize, numSteps, 600, manager)
    val dataset: ArrayDataset = dataNMT.first
    val srcVocab: Vocab = dataNMT.second.first
    val tgtVocab: Vocab = dataNMT.second.second

    val encoder = Seq2SeqEncoder(srcVocab.length(), embedSize, numHiddens, numLayers, dropout)
    val decoder = Seq2SeqAttentionDecoder(tgtVocab.length().toLong(), embedSize, numHiddens, numLayers)

    val net = EncoderDecoder(encoder, decoder)
    fun trainSeq2Seq(
        net: EncoderDecoder,
        dataset: ArrayDataset,
        lr: Float,
        numEpochs: Int,
        tgtVocab: Vocab,
        device: Device
    ) {
        val loss: Loss = MaskedSoftmaxCELoss()
        val lrt: Tracker = Tracker.fixed(lr)
        val adam: Optimizer = Optimizer.adam().optLearningRateTracker(lrt).build()
        val config: DefaultTrainingConfig = DefaultTrainingConfig(loss)
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
                    val X: NDArray = batch.data.get(0)
                    val lenX: NDArray = batch.data.get(1)
                    val Y: NDArray = batch.labels.get(0)
                    val lenY: NDArray = batch.labels.get(1)
                    val bos: NDArray = manager
                        .full(Shape(Y.shape[0]), tgtVocab.getIdx("<bos>"))
                        .reshape(-1, 1)
                    val decInput: NDArray = NDArrays.concat(
                        NDList(bos, Y.get(NDIndex(":, :-1"))),
                        1
                    ) // Teacher forcing
                    Engine.getInstance().newGradientCollector().use { gc ->
                        val yHat: NDArray = net.forward(
                            ParameterStore(manager, false),
                            NDList(X, decInput, lenX),
                            true
                        )
                            .get(0)
                        val l = loss.evaluate(NDList(Y, lenY), NDList(yHat))
                        gc.backward(l)
                        metric.add(floatArrayOf(l.sum().getFloat(), lenY.sum().getLong().toFloat()))
                    }
                    TrainingChapter9.gradClipping(net, 1, manager)
                    // Update parameters
                    trainer.step()
                }
            lossValue = metric.get(0).toDouble() / metric.get(1)
            speed = metric.get(1) / watch.stop()
            if ((epoch + 1) % 10 == 0) {
//            animator.add(epoch + 1, lossValue.toFloat(), "loss")
//            animator.show()
                println("${epoch + 1} : $lossValue")
            }
        }
        println("loss: %.3f, %.1f tokens/sec on %s%n".format(lossValue, speed, device.toString()))
    }
    trainSeq2Seq(net, dataset, lr, numEpochs, tgtVocab, device)

    fun predictSeq2Seq(
        net: EncoderDecoder,
        srcSentence: String,
        srcVocab: Vocab,
        tgtVocab: Vocab,
        numSteps: Int,
        device: Device,
        saveAttentionWeights: Boolean
    ): Pair<String, List<NDArray?>> {
        val srcTokens: List<Int> =
            srcVocab.getIdxs(
                srcSentence.lowercase(Locale.getDefault()).split(" ".toRegex()).dropLastWhile { it.isEmpty() }
            ) + listOf(srcVocab.getIdx("<eos>"))
        val encValidLen = manager.create(srcTokens.size)
        val truncateSrcTokens = NMT.truncatePad(srcTokens, numSteps, srcVocab.getIdx("<pad>"))
        // Add the batch axis
        val encX = manager.create(truncateSrcTokens.toIntArray()).expandDims(0)
        val encOutputs = net.encoder.forward(ParameterStore(manager, false), NDList(encX, encValidLen), false)
        var decState = net.decoder.initState(encOutputs.addAll(NDList(encValidLen)))
        // Add the batch axis
        var decX = manager.create(floatArrayOf(tgtVocab.getIdx("<bos>").toFloat())).expandDims(0)
        val outputSeq: MutableList<Int> = mutableListOf()
        val attentionWeightSeq: MutableList<NDArray?> = mutableListOf()
        for (i in 0 until numSteps) {
            val output = net.decoder.forward(
                ParameterStore(manager, false),
                NDList(decX).addAll(decState),
                false
            )
            val Y = output[0]
            decState = output.subNDList(1)
            // We use the token with the highest prediction likelihood as the input
            // of the decoder at the next time step
            decX = Y.argMax(2)
            val pred = decX.squeeze(0).getLong().toInt()
            // Save attention weights (to be covered later)
            if (saveAttentionWeights) {
                attentionWeightSeq.add(net.decoder.attentionWeights)
            }
            // Once the end-of-sequence token is predicted, the generation of the
            // output sequence is complete
            if (pred == tgtVocab.getIdx("<eos>")) {
                break
            }
            outputSeq.add(pred)
        }
        val outputString: String = " " + tgtVocab.toTokens(outputSeq)
        return Pair(outputString, attentionWeightSeq.toList())
    }

    /* Compute the BLEU. */
    fun bleu(predSeq: String, labelSeq: String, k: Int): Double {
        val predTokens = predSeq.split(" ".toRegex()).dropLastWhile { it.isEmpty() }
        val labelTokens = labelSeq.split(" ".toRegex()).dropLastWhile { it.isEmpty() }
        val lenPred = predTokens.size
        val lenLabel = labelTokens.size
        var score = Math.exp(Math.min(0, 1 - lenLabel / lenPred).toDouble())
        for (n in 1 until k + 1) {
            var numMatches = 0.0
            val labelSubs: MutableMap<String, Int> = mutableMapOf()
            for (i in 0 until lenLabel - n + 1) {
                val key: String = " " + labelTokens.subList(i, i + n)
                labelSubs[key] = labelSubs.getOrDefault(key, 0) + 1
            }
            for (i in 0 until lenPred - n + 1) {
                val key: String = " " + predTokens.subList(i, i + n)
                if (labelSubs.getOrDefault(key, 0) > 0) {
                    numMatches += 1.0
                    labelSubs[key] = labelSubs.getOrDefault(key, 0) - 1
                }
            }
            score *= Math.pow((numMatches / (lenPred - n + 1)).toDouble(), Math.pow(0.5, n.toDouble()))
        }
        return score
    }

    val engs = arrayOf("go .", "i lost .", "he's calm .", "i'm home .")
    val fras = arrayOf("va !", "j'ai perdu .", "il est calme .", "je suis chez moi .")
    for (i in engs.indices) {
        val pair = predictSeq2Seq(net, engs[i], srcVocab, tgtVocab, numSteps, device, false)
        val translation: String = pair.first
        val attentionWeightSeq = pair.second
        println("%s => %s, bleu %.3f".format(engs[i], translation, bleu(translation, fras[i], 2)))
    }

}

abstract class AttentionDecoder : Decoder() {
    override var attentionWeights: NDArray? = null
    val attentionWeightArr : MutableList<NDArray> = mutableListOf()
    abstract override fun initState(encOutputs: NDList): NDList
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }
}

class Seq2SeqAttentionDecoder(
    vocabSize: Long,
    private val embedSize: Int,
    private val numHiddens: Int,
    private val numLayers: Int,
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
        val encValidLens = if(encOutputs.size >= 3) encOutputs[2] else manager.create(0)
        return NDList(outputs.swapAxes(0, 1), hiddenState, encValidLens)
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        embedding.initialize(manager, dataType, inputShapes[0])
        attention.initialize(manager, DataType.FLOAT32, inputShapes[1], inputShapes[1])
        rnn.initialize(manager, DataType.FLOAT32, Shape(1, 4, (numHiddens + embedSize).toLong()))
        linear.initialize(manager, DataType.FLOAT32, Shape(4, numHiddens.toLong()))
    }

    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
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
//            println(attention.attentionWeights?.shape)
//            println(attentionWeights)
            if(attention.attentionWeights != null )  attentionWeightArr.add(attention.attentionWeights!!)
        }
        val ret = linear.forward(ps, NDList(outputs), training)
        return NDList(ret[0].swapAxes(0, 1), encOutputs, hiddenState, encValidLens)
    }
}
