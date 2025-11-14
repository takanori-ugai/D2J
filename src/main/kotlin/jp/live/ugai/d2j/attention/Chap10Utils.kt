package jp.live.ugai.d2j.attention

import ai.djl.Device
import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import jp.live.ugai.d2j.MaskedSoftmaxCELoss
import jp.live.ugai.d2j.lstm.EncoderDecoder
import jp.live.ugai.d2j.manager
import jp.live.ugai.d2j.timemachine.Vocab
import jp.live.ugai.d2j.util.Accumulator
import jp.live.ugai.d2j.util.NMT
import jp.live.ugai.d2j.util.StopWatch
import jp.live.ugai.d2j.util.TrainingChapter9
import java.util.Locale

/**
 * A utility object for Chapter 10 operations.
 */
object Chap10Utils {
    /**
     * Performs a masked softmax operation along the last axis of the input NDArray.
     *
     * @param input The input NDArray (expected 3D tensor).
     * @param validLens NDArray containing valid lengths (1D or 2D), or null for no masking.
     * @return The result NDArray after applying masked softmax.
     */
    fun maskedSoftmax(
        input: NDArray,
        validLens: NDArray?,
    ): NDArray {
        if (validLens == null || validLens.isEmpty) {
            return input.softmax(-1)
        }

        val shape = input.shape
        val lastDim = shape[shape.dimension() - 1]

        // Create a mask based on validLens
        val mask =
            manager
                .arange(lastDim.toFloat())
                .expandDims(0)
                .lt(validLens.reshape(-1, 1))
                .reshape(shape)

        // Apply the mask
        val maskedInput = input.duplicate() // Don't modify the original input
        maskedInput.set(mask.logicalNot(), -1.0E6F)

        return maskedInput.softmax(-1)
    }

    /**
     * Transposes the input NDArray for multi-head attention.
     *
     * Input shape:  (batchSize, numQueriesOrKVs, numHiddens)
     * Output shape: (batchSize * numHeads, numQueriesOrKVs, numHiddens / numHeads)
     *
     * @param input The input NDArray.
     * @param numHeads The number of attention heads.
     * @return The transposed NDArray suitable for multi-head attention.
     */
    fun transposeQkv(
        input: NDArray,
        numHeads: Int,
    ): NDArray {
        val batchSize = input.shape[0]
        val numQueriesOrKVs = input.shape[1]
        val reshaped = input.reshape(batchSize, numQueriesOrKVs, numHeads.toLong(), -1)
        val transposed = reshaped.transpose(0, 2, 1, 3)
        return transposed.reshape(-1, transposed.shape[2], transposed.shape[3])
    }

    /**
     * Transposes the output NDArray back to its original form after multi-head attention.
     *
     * @param _X The input NDArray.
     * @param numHeads The number of attention heads.
     * @return The transposed NDArray.
     */
    fun transposeOutput(
        _X: NDArray,
        numHeads: Int,
    ): NDArray {
        var X = _X
        X = X.reshape(-1, numHeads.toLong(), X.shape[1], X.shape[2])
        X = X.transpose(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    }

    /**
     * Trains a sequence-to-sequence model.
     *
     * @param net The EncoderDecoder network.
     * @param dataset The training dataset.
     * @param lr The learning rate.
     * @param numEpochs The number of training epochs.
     * @param tgtVocab The target vocabulary.
     * @param device The device (CPU/GPU) where the training takes place.
     */
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
        var watch: StopWatch
        var metric: Accumulator
        var lossValue = 0.0
        var speed = 0.0
        for (epoch in 1..numEpochs) {
            watch = StopWatch()
            metric = Accumulator(2) // Sum of training loss, no. of tokens
            for (batch in dataset.getData(manager)) {
                val X: NDArray = batch.data.get(0)
                val lenX: NDArray = batch.data.get(1)
                val Y: NDArray = batch.labels.get(0)
                val lenY: NDArray = batch.labels.get(1)
                val bos: NDArray =
                    manager
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
                TrainingChapter9.gradClipping(net, 1, manager)
                // Update parameters
                trainer.step()
            }
            lossValue = metric.get(0).toDouble() / metric.get(1)
            speed = metric.get(1) / watch.stop()
            if ((epoch + 1) % 10 == 0) {
                println("${epoch + 1} : $lossValue")
            }
        }
        println("loss: %.3f, %.1f tokens/sec on %s%n".format(lossValue, speed, device.toString()))
    }

    /**
     * Tokenizes a sentence, appends <eos>, pads/truncates to numSteps, and returns the NDArray and valid length.
     *
     * @param sentence The input sentence to tokenize.
     * @param vocab The vocabulary for tokenization.
     * @param numSteps The maximum sequence length after padding/truncation.
     * @return Pair of (tokenized & padded NDArray with batch axis, valid length NDArray).
     */
    fun tokenizeAndPad(
        sentence: String,
        vocab: Vocab,
        numSteps: Int,
    ): Pair<NDArray, NDArray> {
        val tokens =
            vocab.getIdxs(sentence.lowercase(Locale.getDefault()).split(" ")) +
                listOf(vocab.getIdx("<eos>"))
        val validLen = manager.create(tokens.size)
        val paddedTokens = NMT.truncatePad(tokens, numSteps, vocab.getIdx("<pad>"))
        val tokenArray = manager.create(paddedTokens.toIntArray()).expandDims(0)
        return Pair(tokenArray, validLen)
    }

    /**
     * Makes predictions with a sequence-to-sequence model.
     *
     * @param net The EncoderDecoder network.
     * @param srcSentence The source sentence.
     * @param srcVocab The source vocabulary.
     * @param tgtVocab The target vocabulary.
     * @param numSteps The number of time steps in the prediction.
     * @param device The device (CPU/GPU) where the prediction takes place.
     * @param saveAttentionWeights Whether to save the attention weights.
     * @return The predicted sequence and the attention weights.
     */
    fun predictSeq2Seq(
        net: EncoderDecoder,
        srcSentence: String,
        srcVocab: Vocab,
        tgtVocab: Vocab,
        numSteps: Int,
        device: Device,
        saveAttentionWeights: Boolean,
    ): Pair<String, List<List<Pair<FloatArray, Shape>>>> {
        val (encX, encValidLen) = tokenizeAndPad(srcSentence, srcVocab, numSteps)
        val encOutputs = net.encoder.forward(ParameterStore(manager, false), NDList(encX, encValidLen), false)
        var decState = net.decoder.initState(encOutputs.addAll(NDList(encValidLen)))
        // Add the batch axis
        var decX = manager.create(floatArrayOf(tgtVocab.getIdx("<bos>").toFloat())).expandDims(0)
        val outputSeq: MutableList<Int> = mutableListOf()
        val attentionWeightSeq: MutableList<List<Pair<FloatArray, Shape>>> = mutableListOf()
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
                attentionWeightSeq.add((net.decoder as AttentionDecoder).attentionWeightArr)
            }
            // Once the end-of-sequence token is predicted, the generation of the
            // output sequence is complete
            if (pred == tgtVocab.getIdx("<eos>")) {
                break
            }
            outputSeq.add(pred)
        }
        val outputString: String = tgtVocab.toTokens(outputSeq).joinToString(separator = " ")
        return Pair(outputString, attentionWeightSeq)
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
        val predTokens = predSeq.split(" ").filter { it.isNotEmpty() }
        val labelTokens = labelSeq.split(" ").filter { it.isNotEmpty() }
        val predLen = predTokens.size
        val labelLen = labelTokens.size

        if (predLen == 0 || labelLen == 0) {
            return 0.0
        }

        val score = Math.exp(Math.min(0.0, 1.0 - labelLen.toDouble() / predLen))

        val precisions = DoubleArray(maxOrder)
        for (n in 1..maxOrder) {
            val predNgrams = mutableMapOf<String, Int>()
            for (i in 0..predLen - n) {
                val ngram = predTokens.subList(i, i + n).joinToString(" ")
                predNgrams[ngram] = predNgrams.getOrDefault(ngram, 0) + 1
            }

            val labelNgrams = mutableMapOf<String, Int>()
            for (i in 0..labelLen - n) {
                val ngram = labelTokens.subList(i, i + n).joinToString(" ")
                labelNgrams[ngram] = labelNgrams.getOrDefault(ngram, 0) + 1
            }

            var clippedCount = 0
            for ((ngram, count) in predNgrams) {
                clippedCount += minOf(count, labelNgrams.getOrDefault(ngram, 0))
            }

            val totalCount = predLen - n + 1
            precisions[n - 1] = if (totalCount > 0) clippedCount.toDouble() / totalCount else 0.0
        }

        val geometricMean =
            if (precisions.minOrNull()!! == 0.0) {
                0.0
            } else {
                val sumOfLogs = precisions.map { Math.log(it) }.sum()
                Math.exp(sumOfLogs / maxOrder)
            }

        return score * geometricMean
    }
}
