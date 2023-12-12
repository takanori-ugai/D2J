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
     * Performs softmax operation by masking elements on the last axis.
     *
     * @param _X The input NDArray.
     * @param _validLens The valid lengths of the input.
     * @return The result NDArray after masked softmax operation.
     */
    fun maskedSoftmax(
        _X: NDArray,
        _validLens: NDArray?,
    ): NDArray {
        // Perform softmax operation by masking elements on the last axis.
        // `X`: 3D tensor, `validLens`: 1D or 2D tensor
        val shape: Shape = _X.shape
        var validLens = _validLens
        if (_validLens == null) {
            return _X.reshape(Shape(-1, shape.get(shape.dimension() - 1))).softmax(-1).reshape(shape)
        }
        if (validLens!!.shape.dimension() == 0) {
            return _X.softmax(-1).reshape(shape)
        }
        validLens =
            if (validLens.shape.dimension() == 1) {
                validLens.repeat(shape.get(1))
            } else {
                validLens.reshape(-1)
            }
        // On the last axis, replace masked elements with a very large negative
        // value, whose exponentiation outputs 0
        return _X.reshape(Shape(-1, shape.get(shape.dimension() - 1)))
            .sequenceMask(validLens, -1.0E6F)
            .softmax(-1)
            .reshape(shape)
    }

    /**
     * Transposes the input NDArray for multi-head attention.
     *
     * @param _X The input NDArray.
     * @param numHeads The number of attention heads.
     * @return The transposed NDArray.
     */
    fun transposeQkv(
        _X: NDArray,
        numHeads: Int,
    ): NDArray {
        // Shape of input `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHiddens`).
        // Shape of output `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHeads`,
        // `numHiddens` / `numHeads`)
        var X = _X
        X = X.reshape(X.shape[0], X.shape[1], numHeads.toLong(), -1)

        // Shape of output `X`:
        // (`batchSize`, `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        X = X.transpose(0, 2, 1, 3)

        // Shape of `output`:
        // (`batchSize` * `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        return X.reshape(-1, X.shape[2], X.shape[3])
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
                        net.forward(
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

    fun tokenizeAndPad(
        srcSentence: String,
        srcVocab: Vocab,
        numSteps: Int,
    ): Pair<NDArray, NDArray> {
        val srcTokens = srcVocab.getIdxs(srcSentence.lowercase(Locale.getDefault()).split(" ")) + listOf(srcVocab.getIdx("<eos>"))
        val encValidLen = manager.create(srcTokens.size)
        val truncateSrcTokens = NMT.truncatePad(srcTokens, numSteps, srcVocab.getIdx("<pad>"))
        // Add the batch axis
        val encX = manager.create(truncateSrcTokens.toIntArray()).expandDims(0)
        return Pair(encX, encValidLen)
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
     * @param predSeq The predicted sequence.
     * @param labelSeq The ground truth sequence.
     * @param k The maximum order of n-grams for which matching statistics are computed.
     * @return The BLEU score.
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
        var score = Math.exp(Math.min(0.0, 1.0 - lenLabel / lenPred))
        for (n in 1..k) {
            var numMatches = 0
            val labelSubs = mutableMapOf<String, Int>()
            for (i in 0 until lenLabel - n + 1) {
                val key = labelTokens.subList(i, i + n).joinToString(separator = " ")
                labelSubs.put(key, labelSubs.getOrDefault(key, 0) + 1)
            }
            for (i in 0 until lenPred - n + 1) {
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
}
