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
import ai.djl.nn.recurrent.GRU
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.loss.Loss
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import jp.live.ugai.d2j.lstm.Encoder
import jp.live.ugai.d2j.lstm.EncoderDecoder
import jp.live.ugai.d2j.timemachine.Vocab
import jp.live.ugai.d2j.util.Accumulator
import jp.live.ugai.d2j.util.NMT
import jp.live.ugai.d2j.util.NMT.loadDataNMT
import jp.live.ugai.d2j.util.StopWatch
import jp.live.ugai.d2j.util.TrainingChapter9.gradClipping
import java.util.Locale

fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    val manager = NDManager.newBaseManager()
    val ps = ParameterStore(manager, false)
    var encoder = Seq2SeqEncoder(10, 8, 16, 2, 0.0f)
    var X = manager.zeros(Shape(4, 7))
    encoder.initialize(manager, DataType.FLOAT32, X.shape)
    var outputState = encoder.forward(ps, NDList(X), false)
    var output = outputState.head()

    println(output.shape)

    var state = outputState.subNDList(1)
    println(state.size)
    println(state.head().shape)

    var decoder = Seq2SeqDecoder(10, 8, 16, 2, 0f)
    state = decoder.initState(outputState)
    val input = NDList(X).addAll(state)
    decoder.initialize(manager, DataType.FLOAT32, *input.shapes)
    outputState = decoder.forward(ps, input, false)

    output = outputState.head()
    println(output.shape)

    state = outputState.subNDList(1)
    println(state.size)
    println(state.head().shape)

    X = manager.create(arrayOf(intArrayOf(1, 2, 3), intArrayOf(4, 5, 6)))
    println(X.sequenceMask(manager.create(intArrayOf(1, 2))))

    X = manager.ones(Shape(2, 3, 4))
    println(X.sequenceMask(manager.create(intArrayOf(1, 2)), -1f))

    val loss: Loss = MaskedSoftmaxCELoss()
    val labels = NDList(manager.ones(Shape(3, 4)))
    labels.add(manager.create(intArrayOf(4, 2, 0)))
    val predictions = NDList(manager.ones(Shape(3, 4, 10)))
    println(loss.evaluate(labels, predictions))

    fun trainSeq2Seq(
        net: EncoderDecoder,
        dataset: ArrayDataset,
        lr: Float,
        numEpochs: Int,
        tgtVocab: Vocab,
        device: Device,
    ) {
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
            manager.newSubManager(device).use { childManager ->
                // Iterate over dataset
                for (batch in dataset.getData(childManager)) {
                    val X: NDArray = batch.data.get(0)
                    val lenX: NDArray = batch.data.get(1)
                    val Y: NDArray = batch.labels.get(0)
                    val lenY: NDArray = batch.labels.get(1)
                    val bos: NDArray =
                        childManager
                            .full(Shape(Y.shape[0]), tgtVocab.getIdx("<bos>"))
                            .reshape(-1, 1)
                    val decInput: NDArray =
                        NDArrays.concat(
                            NDList(bos, Y.get(NDIndex(":, :-1"))),
                            1,
                        ) // Teacher forcing
                    Engine.getInstance().newGradientCollector().use { gc ->
                        val yHat: NDArray =
                            net
                                .forward(
                                    ParameterStore(manager, false),
                                    NDList(X, decInput, lenX),
                                    true,
                                ).get(0)
                        val l = loss.evaluate(NDList(Y, lenY), NDList(yHat))
                        gc.backward(l)
                        metric.add(floatArrayOf(l.sum().getFloat(), lenY.sum().getLong().toFloat()))
                    }
                    gradClipping(net, 1, childManager)
                    // Update parameters
                    trainer.step()
                }
            }
            lossValue = metric.get(0).toDouble() / metric.get(1)
            speed = metric.get(1) / watch.stop()
            if ((epoch + 1) % 10 == 0) {
//            animator.add(epoch + 1, lossValue.toFloat(), "loss")
//            animator.show()
                println("${epoch + 1} : $lossValue")
            }
        }
        println(
            "loss: %.3f, %.1f tokens/sec on %s%n".format(lossValue, speed, device.toString()),
        )
    }

    val embedSize = 32
    val numHiddens = 32
    val numLayers = 2
    val batchSize = 64
    val numSteps = 10
    val numEpochs = Integer.getInteger("MAX_EPOCH", 30)

    val dropout = 0.1f
    val lr = 0.005f
    val device = manager.device

    val dataNMT = loadDataNMT(batchSize, numSteps, 600)
    val dataset: ArrayDataset = dataNMT.first
    val srcVocab: Vocab = dataNMT.second.first
    val tgtVocab: Vocab = dataNMT.second.second
    println("Target: ${manager.create(floatArrayOf(tgtVocab.getIdx("<bos>").toFloat())).expandDims(0)}")

    encoder = Seq2SeqEncoder(srcVocab.length(), embedSize, numHiddens, numLayers, dropout)
    decoder = Seq2SeqDecoder(tgtVocab.length(), embedSize, numHiddens, numLayers, dropout)

    val net = EncoderDecoder(encoder, decoder)
    trainSeq2Seq(net, dataset, lr, numEpochs, tgtVocab, device)

    fun predictSeq2Seq(
        net: EncoderDecoder,
        srcSentence: String,
        srcVocab: Vocab,
        tgtVocab: Vocab,
        numSteps: Int,
        device: Device,
        saveAttentionWeights: Boolean,
    ): Pair<String, List<NDArray?>> {
        val srcTokens = srcVocab.getIdxs(srcSentence.lowercase(Locale.getDefault()).split(" ")) + listOf(srcVocab.getIdx("<eos>"))
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
            val output =
                net.decoder.forward(
                    ParameterStore(manager, false),
                    NDList(decX).addAll(decState),
                    false,
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
        val outputString: String = tgtVocab.toTokens(outputSeq).joinToString(separator = " ")
        return Pair(outputString, attentionWeightSeq.toList())
    }

    /**
     * Calculates the BLEU score for a predicted sequence.
     *
     * @param predSeq The predicted sequence as a space-separated string.
     * @param labelSeq The ground truth sequence as a space-separated string.
     * @param maxOrder The maximum order of n-grams for which matching statistics are computed.
     * @return The BLEU score.
     */
    fun bleu(
        predSeq: String,
        labelSeq: String,
        maxOrder: Int,
    ): Double {
        val predTokens = predSeq.split(" ")
        val labelTokens = labelSeq.split(" ")
        val predLen = predTokens.size
        val labelLen = labelTokens.size

        if (predLen == 0 || labelLen == 0) return 0.0

        // Brevity penalty
        var score = Math.exp(Math.min(0.0, 1.0 - labelLen.toDouble() / predLen))

        for (n in 1..maxOrder) {
            var matchCount = 0
            val labelNgrams = mutableMapOf<String, Int>()

            for (i in 0..labelLen - n) {
                val ngram = labelTokens.subList(i, i + n).joinToString(" ")
                labelNgrams[ngram] = labelNgrams.getOrDefault(ngram, 0) + 1
            }

            for (i in 0..predLen - n) {
                val ngram = predTokens.subList(i, i + n).joinToString(" ")
                val count = labelNgrams.getOrDefault(ngram, 0)
                if (count > 0) {
                    matchCount++
                    labelNgrams[ngram] = count - 1
                }
            }

            val possibleMatches = predLen - n + 1
            if (possibleMatches <= 0) return 0.0

            score *= Math.pow(matchCount.toDouble() / possibleMatches, Math.pow(0.5, n.toDouble()))
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

class Seq2SeqEncoder(
    vocabSize: Int,
    embedSize: Int,
    numHiddens: Int,
    numLayers: Int,
    dropout: Float,
) : Encoder() {
    private val embedding: TrainableWordEmbedding
    private val rnn: GRU

    // The RNN encoder for sequence to sequence learning.
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

    /** {@inheritDoc}  */
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

    override fun forwardInternal(
        ps: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        var X = inputs.head()
        // The output `X` shape: (`batchSize`, `numSteps`, `embedSize`)
        X = embedding.forward(ps, NDList(X), training, params).head()
        // In RNN models, the first axis corresponds to time steps
        X = X.swapAxes(0, 1)
        return rnn.forward(ps, NDList(X), training)
    }
}

class MaskedSoftmaxCELoss : SoftmaxCrossEntropyLoss() {
    // The softmax cross-entropy loss with masks.
    override fun evaluate(
        labels: NDList,
        predictions: NDList,
    ): NDArray {
        val weights =
            labels
                .head()
                .onesLike()
                .expandDims(-1)
                .sequenceMask(labels[1])
        // Remove the states from the labels NDList because otherwise, it will throw an error as SoftmaxCrossEntropyLoss
        // expects only one NDArray for label and one NDArray for prediction
        labels.removeAt(1)
        return super.evaluate(labels, predictions).mul(weights).mean(intArrayOf(1))
    }
}
