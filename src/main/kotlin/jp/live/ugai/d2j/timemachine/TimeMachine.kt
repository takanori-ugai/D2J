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
import jp.live.ugai.d2j.util.Functions
import jp.live.ugai.d2j.util.StopWatch
import jp.live.ugai.d2j.util.Training.sgd
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.URL

object TimeMachine {
    /** Split text lines into word or character tokens.  */
    fun tokenize(lines: List<String>, token: String): List<List<String>> {
        val output: MutableList<List<String>> = mutableListOf()
        if ("word" == token) {
            for (line in lines) {
                output.add(line.split(" ".toRegex()).dropLastWhile { it.isEmpty() })
            }
        } else if ("char" == token) {
            for (line in lines) {
                output.add(line.split("".toRegex()).dropLastWhile { it.isEmpty() })
            }
        } else {
            throw IllegalArgumentException("ERROR: unknown token type: $token")
        }
        return output
    }

    /** Read `The Time Machine` dataset and return an array of the lines  */
    fun readTimeMachine(): List<String> {
        val url = URL("http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt")
        var lines: List<String> = mutableListOf()
        BufferedReader(InputStreamReader(url.openStream())).use { inp ->
            lines = inp.readLines()
        }
        val retLines = mutableListOf<String>()
        for (line in lines) {
            retLines.add(line.replace("[^A-Za-z]+".toRegex(), " ").trim().lowercase(java.util.Locale.getDefault()))
        }
        return retLines
    }

    /** Return token indices and the vocabulary of the time machine dataset.  */
    fun loadCorpusTimeMachine(maxTokens: Int): Pair<List<Int>, Vocab> {
        val lines = readTimeMachine()
        val tokens = tokenize(lines, "char")
        val vocab = Vocab(tokens, 0, listOf<String>())
        // Since each text line in the time machine dataset is not necessarily a
        // sentence or a paragraph, flatten all the text lines into a single list
        var corpus: MutableList<Int> = mutableListOf()
        for (token in tokens) {
            for (s in token) {
                if (!s.isEmpty()) {
                    corpus.add(vocab.getIdx(s))
                }
            }
        }
        if (maxTokens > 0) {
            corpus = corpus.subList(0, maxTokens)
        }
        return Pair(corpus, vocab)
    }

    /** Generate new characters following the `prefix`.  */
    fun predictCh8(
        prefix: String,
        numPreds: Int,
        net: Any,
        vocab: Vocab,
        device: Device,
        manager: NDManager
    ): String {
        val outputs: MutableList<Int> = mutableListOf()
        outputs.add(vocab.getIdx("" + prefix[0]))
        val getInput = {
            manager.create(outputs[outputs.size - 1])
                .toDevice(device, false)
                .reshape(Shape(1, 1))
        }
        if (net is RNNModelScratch) {
            val castedNet = net
            var state: NDList = castedNet.beginState(1, device)
            for (c in prefix.substring(1).toCharArray()) { // Warm-up period
                state = castedNet.forward(getInput(), state).second
                outputs.add(vocab.getIdx("" + c))
            }
            for (i in 0 until numPreds) {
                val pair = castedNet.forward(getInput(), state)
                val y = pair.first
                state = pair.second
                outputs.add(y.argMax(1).reshape(Shape(1)).getLong(0L).toInt())
            }
        } else {
            val castedNet = net as AbstractBlock
            var state: NDList? = null
            for (c in prefix.substring(1).toCharArray()) { // Warm-up period
                val ps = ParameterStore(manager, false)
                val input = NDList(getInput())
                state = if (state == null) {
                    // Begin state
                    castedNet.forward(ps, input, false).subNDList(1)
                } else {
                    castedNet.forward(ps, input.addAll(state), false).subNDList(1)
                }
                outputs.add(vocab.getIdx("" + c))
            }
            for (i in 0 until numPreds) {
                val ps = ParameterStore(manager, false)
                val input = NDList(getInput())
                val pair = castedNet.forward(ps, input.addAll(state), false)
                val y = pair[0]
                state = pair.subNDList(1)
                outputs.add(y.argMax(1).reshape(Shape(1)).getLong(0L).toInt())
            }
        }
        val output = StringBuilder()
        for (i in outputs) {
            output.append(vocab.idxToToken[i])
        }
        return output.toString()
    }

    /** Train a model.  */
    fun trainCh8(
        net: Any,
        dataset: RandomAccessDataset,
        vocab: Vocab,
        lr: Int,
        numEpochs: Int,
        device: Device,
        useRandomIter: Boolean,
        manager: NDManager
    ) {
        val loss = SoftmaxCrossEntropyLoss()
//        val animator = Animator()
        val updater: (Int, NDManager) -> Unit
        if (net is RNNModelScratch) {
            updater = { batchSize: Int, subManager: NDManager ->
                sgd(
                    net.params,
                    lr.toFloat(),
                    batchSize,
                    subManager
                )
            }
        } else {
            // Already initialized net
            val castedNet = net as AbstractBlock
            val model = Model.newInstance("model")
            model.block = castedNet
            val lrt = Tracker.fixed(lr.toFloat())
            val sgd: Optimizer = Optimizer.sgd().setLearningRateTracker(lrt).build()
            val config = DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer (loss function)
                .optInitializer(
                    NormalInitializer(0.01f),
                    Parameter.Type.WEIGHT
                ) // setting the initializer
                .optDevices(
                    manager.engine
                        .getDevices(1)
                ) // setting the number of GPUs needed
                .addEvaluator(Accuracy()) // Model Accuracy
                .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging
            val trainer = model.newTrainer(config)
            updater = { batchSize: Int, subManager: NDManager -> trainer.step() }
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
//            animator.add(epoch + 1, ppl.toFloat(), "")
//            animator.show()
                println("${epoch + 1} : $ppl")
            }
        }
        println(
            "perplexity: %.1f, %.1f tokens/sec on %s%n".format(ppl, speed, device.toString())
        )
        println(predict("time traveller"))
        println(predict("traveller"))
    }

    fun trainEpochCh8(
        net: Any,
        dataset: RandomAccessDataset,
        loss: Loss,
        updater: (Int, NDManager) -> Unit,
        device: Device,
        useRandomIter: Boolean,
        manager: NDManager
    ): Pair<Double, Double> {
        val watch = StopWatch()
        watch.start()
        val metric = Accumulator(2) // Sum of training loss, no. of tokens
        manager.newSubManager().use { childManager ->
            var state: NDList? = null
            for (batch in dataset.getData(manager)) {
                var X = batch.data.head().toDevice(Functions.tryGpu(0), true)
                X.attach(childManager)
                val Y = batch.labels.head().toDevice(Functions.tryGpu(0), true)
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
                var y = Y.transpose().reshape(Shape(-1))
                X = X.toDevice(device, false)
                y = y.toDevice(device, false)
                Engine.getInstance().newGradientCollector().use { gc ->
                    val yHat: NDArray
//                    println(state)
                    if (net is RNNModelScratch) {
                        val pairResult = net.forward(X, state!!)
                        yHat = pairResult.first
                        state = pairResult.second
                    } else {
                        val pairResult: NDList
                        pairResult = if (state == null) {
                            // Begin state
                            (net as AbstractBlock)
                                .forward(
                                    ParameterStore(manager, false),
                                    NDList(X),
                                    true
                                )
                        } else {
                            (net as AbstractBlock)
                                .forward(
                                    ParameterStore(manager, false),
                                    NDList(X).addAll(state),
                                    true
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

    /** Clip the gradient.  */
    fun gradClipping(net: Any, theta: Int, manager: NDManager?) {
        var result = 0.0
        val params: NDList
        if (net is RNNModelScratch) {
            params = net.params
        } else {
            params = NDList()
            for (pair in (net as AbstractBlock).parameters) {
                params.add(pair.value.array)
            }
        }
        for (p in params) {
            val gradient = p.gradient.stopGradient()
            gradient.attach(manager)
            result += gradient.pow(2).sum().getFloat().toDouble()
        }
        val norm = Math.sqrt(result)
        if (norm > theta) {
            for (param in params) {
                val gradient = param.gradient
                gradient.muli(theta / norm)
            }
        }
    }
}
