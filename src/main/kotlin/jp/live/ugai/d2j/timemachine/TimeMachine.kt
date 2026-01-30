package jp.live.ugai.d2j.timemachine

import ai.djl.Device
import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.dataset.RandomAccessDataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import jp.live.ugai.d2j.util.Accumulator
import jp.live.ugai.d2j.util.StopWatch
import jp.live.ugai.d2j.util.Training.sgd
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.URI

/**
 * Singleton for TimeMachine.
 */
object TimeMachine {
    /**
     * Splits text lines into tokens based on the specified tokenization strategy.
     *
     * @param lines The input text lines to tokenize.
     * @param token The tokenization mode: "word" for whitespace splitting, "char" for character splitting.
     * @return A list of token lists, one per input line.
     */
    fun tokenize(
        lines: List<String>,
        token: String,
    ): List<List<String>> =
        when (token) {
            "word" -> lines.map { it.split(" ".toRegex()).filter { word -> word.isNotEmpty() } }
            "char" -> lines.map { it.split("".toRegex()).filter { char -> char.isNotEmpty() } }
            else -> throw IllegalArgumentException("ERROR: unknown token type: $token")
        }

    /**
     * Reads the Time Machine dataset lines from the remote source.
     *
     * @return The raw text lines from the dataset.
     */
    fun readTimeMachine(): List<String> {
        val url = URI("http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt").toURL()
        return BufferedReader(InputStreamReader(url.openStream())).use { inp ->
            inp.readLines().map { line ->
                line.replace("[^A-Za-z]+".toRegex(), " ").trim().lowercase(java.util.Locale.getDefault())
            }
        }
    }

    /**
     * Loads and tokenizes the Time Machine corpus, returning token ids and vocabulary.
     *
     * @param maxTokens The maximum number of tokens to keep (0 or less for no limit).
     * @return A pair of token id list and vocabulary.
     */
    fun loadCorpusTimeMachine(maxTokens: Int): Pair<List<Int>, Vocab> {
        val lines = readTimeMachine()
        val tokens = tokenize(lines, "char")
        val vocab = Vocab(tokens, 0, listOf<String>())
        // Since each text line in the time machine dataset is not necessarily a
        // sentence or a paragraph, flatten all the text lines into a single list
        var corpus =
            tokens.flatMap { token ->
                token.filter { it.isNotEmpty() }.map { vocab.getIdx(it) }
            }
        if (maxTokens > 0 && corpus.size > maxTokens) {
            corpus = corpus.subList(0, maxTokens)
        }
        return Pair(corpus, vocab)
    }

    /**
     * Generates text by predicting the next tokens from a prefix.
     *
     * @param prefix The seed text to start generation.
     * @param numPreds The number of tokens to generate.
     * @param net The model (scratch or concise) used for prediction.
     * @param vocab The vocabulary for token conversions.
     * @param device The device to run inference on.
     * @param manager The NDManager used for array creation.
     * @return The generated text.
     */
    fun predictCh8(
        prefix: String,
        numPreds: Int,
        net: Any,
        vocab: Vocab,
        device: Device,
        manager: NDManager,
    ): String {
        val outputs = mutableListOf(vocab.getIdx("" + prefix[0]))
        val getInput = {
            manager
                .create(outputs[outputs.size - 1])
                .toDevice(device, false)
                .reshape(Shape(1, 1))
        }
        when (net) {
            is RNNModelScratch -> {
                var state: NDList = net.beginState(1, device)
                for (c in prefix.substring(1)) { // Warm-up period
                    state = net.forward(getInput(), state).second
                    outputs.add(vocab.getIdx("" + c))
                }
                repeat(numPreds) {
                    val pair = net.forward(getInput(), state)
                    val y = pair.first
                    state = pair.second
                    outputs.add(
                        y
                            .argMax(1)
                            .reshape(Shape(1))
                            .getLong(0L)
                            .toInt(),
                    )
                }
            }

            is AbstractBlock -> {
                var state: NDList? = null
                for (c in prefix.substring(1)) { // Warm-up period
                    val ps = ParameterStore(manager, false)
                    val input = NDList(getInput())
                    state =
                        if (state == null) {
                            // Begin state
                            net.forward(ps, input, false).subNDList(1)
                        } else {
                            net.forward(ps, input.addAll(state), false).subNDList(1)
                        }
                    outputs.add(vocab.getIdx("" + c))
                }
                repeat(numPreds) {
                    val ps = ParameterStore(manager, false)
                    val input = NDList(getInput())
                    val pair = net.forward(ps, input.addAll(state), false)
                    val y = pair[0]
                    state = pair.subNDList(1)
                    outputs.add(
                        y
                            .argMax(1)
                            .reshape(Shape(1))
                            .getLong(0L)
                            .toInt(),
                    )
                }
            }

            else -> {
                throw IllegalArgumentException("Unsupported network type: ${net::class.simpleName}")
            }
        }

        return outputs.joinToString("") { vocab.idxToToken[it] }
    }

    /**
     * Trains a language model for a number of epochs and prints periodic metrics.
     *
     * @param net The model to train.
     * @param dataset The training dataset.
     * @param vocab The vocabulary for decoding predictions.
     * @param lr The learning rate.
     * @param numEpochs The number of training epochs.
     * @param device The device to train on.
     * @param useRandomIter Whether to use random sampling in the data iterator.
     * @param manager The NDManager for array allocation.
     */
    fun trainCh8(
        net: Any,
        dataset: RandomAccessDataset,
        vocab: Vocab,
        lr: Int,
        numEpochs: Int,
        device: Device,
        useRandomIter: Boolean,
        manager: NDManager,
    ) {
        val loss = SoftmaxCrossEntropyLoss()
//        val animator = Animator()
        val updater: (Int, NDManager) -> Unit =
            when (net) {
                is RNNModelScratch -> { batchSize: Int, subManager: NDManager ->
                    sgd(net.params, lr.toFloat(), batchSize, subManager)
                }

                is AbstractBlock -> { batchSize: Int, subManager: NDManager ->
                    val model = Model.newInstance("model")
                    model.block = net
                    val lrt = Tracker.fixed(lr.toFloat())
                    val sgd: Optimizer = Optimizer.sgd().setLearningRateTracker(lrt).build()
                    val config =
                        DefaultTrainingConfig(loss)
                            .optOptimizer(sgd) // Optimizer (loss function)
                            .optInitializer(
                                NormalInitializer(0.01f),
                                Parameter.Type.WEIGHT,
                            ) // setting the initializer
                            .optDevices(
                                manager.engine
                                    .getDevices(1),
                            ) // setting the number of GPUs needed
                            .addEvaluator(Accuracy()) // Model Accuracy
                            .also { cfg ->
                                TrainingListener.Defaults.logging().forEach { cfg.addTrainingListeners(it) }
                            } // Logging
                    model.newTrainer(config).step()
                }

                else -> throw IllegalArgumentException("Unsupported network type: ${net::class.simpleName}")
            }

        val predict = { prefix: String -> predictCh8(prefix, 50, net, vocab, device, manager) }
        // Train and predict
        var ppl = 0.0
        var speed = 0.0
        for (epoch in 0 until numEpochs) {
            val pair = trainEpochCh8(net, dataset, loss, updater, device, useRandomIter, manager)
            ppl = pair.first
            speed = pair.second
            if ((epoch + 1) % 10 == 0) {
                println("${epoch + 1} : $ppl")
            }
        }
        println(
            "perplexity: %.1f, %.1f tokens/sec on %s%n".format(ppl, speed, device.toString()),
        )
        println(predict("time traveller"))
        println(predict("traveller"))
    }

    /**
     * Trains the model for a single epoch and returns perplexity and speed.
     *
     * @param net The model to train.
     * @param dataset The training dataset.
     * @param loss The loss function.
     * @param updater Parameter update function.
     * @param device The device to train on.
     * @param useRandomIter Whether to use random sampling in the data iterator.
     * @param manager The NDManager for array allocation.
     * @return A pair of (perplexity, tokens per second).
     */
    fun trainEpochCh8(
        net: Any,
        dataset: RandomAccessDataset,
        loss: Loss,
        updater: (Int, NDManager) -> Unit,
        device: Device,
        useRandomIter: Boolean,
        manager: NDManager,
    ): Pair<Double, Double> {
        val watch = StopWatch()
        watch.start()
        val metric = Accumulator(2) // Sum of training loss, no. of tokens
        manager.newSubManager().use { childManager ->
            var state: NDList? = null
            for (batch in dataset.getData(childManager)) {
                var X = batch.data.head().toDevice(device, true)
                X.attach(childManager)
                val Y = batch.labels.head().toDevice(device, true)
                Y.attach(childManager)
                if (state == null || useRandomIter) {
                    // Initialize `state` when either it is the first iteration or
                    // using random sampling
                    if (net is RNNModelScratch) {
                        state = net.beginState(X.shape.shape[0].toInt(), device)
                    }
                } else {
                    for (s in state) {
                        s.stopGradient()
                    }
                }
                state?.attach(childManager)
                val y = Y.transpose().reshape(Shape(-1)).toDevice(device, false)
                X = X.toDevice(device, false)
                Engine.getInstance().newGradientCollector().use { gc ->
                    val yHat: NDArray
//                    println(state)
                    if (net is RNNModelScratch) {
                        val pairResult = net.forward(X, state!!)
                        yHat = pairResult.first
                        state = pairResult.second
                    } else {
                        val pairResult: NDList =
                            if (state == null) {
                                // Begin state
                                (net as AbstractBlock)
                                    .forward(
                                        ParameterStore(childManager, false),
                                        NDList(X),
                                        true,
                                    )
                            } else {
                                (net as AbstractBlock)
                                    .forward(
                                        ParameterStore(childManager, false),
                                        NDList(X).addAll(state),
                                        true,
                                    )
                            }
                        yHat = pairResult[0]
                        state = pairResult.subNDList(1)
                    }
                    val l = loss.evaluate(NDList(y), NDList(yHat)).mean()
                    gc.backward(l)
                    metric.add(floatArrayOf(l.getFloat() * y.size(), y.size().toFloat()))
                }
                gradClipping(net, 1, childManager)
                updater(1, childManager) // Since the `mean` function has been invoked
            }
        }
        return Pair(Math.exp((metric.get(0) / metric.get(1)).toDouble()), metric.get(1) / watch.stop())
    }

    /**
     * Clips gradients to mitigate exploding gradients.
     *
     * @param net The model whose gradients will be clipped.
     * @param theta The clipping threshold.
     * @param manager The NDManager for gradient operations.
     */
    fun gradClipping(
        net: Any,
        theta: Int,
        manager: NDManager?,
    ) {
        val params: NDList =
            when (net) {
                is RNNModelScratch -> net.params
                is AbstractBlock -> NDList().apply { net.parameters.forEach { add(it.value.array) } }
                else -> throw IllegalArgumentException("Unsupported network type: ${net::class.simpleName}")
            }
        val norm =
            Math.sqrt(
                params.sumOf { p ->
                    val gradient = p.gradient.stopGradient()
                    gradient.attach(manager)
                    gradient
                        .pow(2)
                        .sum()
                        .getFloat()
                        .toDouble()
                },
            )
        if (norm > theta) {
            for (param in params) {
                val gradient = param.gradient
                gradient.muli(theta / norm)
            }
        }
    }
}
