package jp.live.ugai.d2j

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.loss.Loss
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import jp.live.ugai.d2j.timemachine.RNNModelScratch
import jp.live.ugai.d2j.timemachine.Vocab
import jp.live.ugai.d2j.util.Accumulator
import jp.live.ugai.d2j.util.StopWatch
import jp.live.ugai.d2j.util.Training.sgd

/**
 * Executes main.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val batchSize = 32
        val numSteps = 35
        val timeMachine: Pair<List<NDList>, Vocab> =
            SeqDataLoader.loadDataTimeMachine(batchSize, numSteps, false, 10000)
        val trainIter: List<NDList> = timeMachine.first
        val vocab: Vocab = timeMachine.second

        println(vocab.length())
        println(manager.create(intArrayOf(0, 2)).oneHot(vocab.length()))
        var sampleData = manager.arange(10).reshape(Shape(2, 5))
        println(sampleData.transpose().oneHot(28).shape)

        val numHiddens = 512
        val getParamsFn = { vocabSize: Int, hiddenSize: Int, device: Device ->
            getParams(manager, vocabSize, hiddenSize, device)
        }
        val initRNNStateFn = { batchSizeValue: Int, hiddenSize: Int, device: Device ->
            initRNNState(manager, batchSizeValue, hiddenSize, device)
        }
        val rnnFn = ::rnn

        sampleData = manager.arange(10).reshape(Shape(2, 5))
        val device = manager.device
        val net = RNNModelScratch(vocab.length(), numHiddens, device, getParamsFn, initRNNStateFn, rnnFn)
        val state = net.beginState(sampleData.shape.shape[0].toInt(), device)
        val pairResult: Pair<NDArray, NDList> = net.forward(sampleData.toDevice(device, false), state)
        val output: NDArray = pairResult.first
        val newState: NDList = pairResult.second
        println(output.shape)
        println(newState[0].shape)

        println(predictCh8(manager, "time traveller ", 10, net, vocab, manager.device))

        val numEpochs = Integer.getInteger("MAX_EPOCH", 500)

        val lr = 1
        trainCh8(manager, net, trainIter, vocab, lr, numEpochs, manager.device, false)
    }
}

/**
 * Executes getParams.
 */
fun getParams(
    manager: NDManager,
    vocabSize: Int,
    numHiddens: Int,
    device: Device,
): NDList {
    // Hidden layer parameters
    val weightXh: NDArray = normal(manager, Shape(vocabSize.toLong(), numHiddens.toLong()), device)
    val weightHh: NDArray = normal(manager, Shape(numHiddens.toLong(), numHiddens.toLong()), device)
    val biasH: NDArray = manager.zeros(Shape(numHiddens.toLong()), DataType.FLOAT32, device)
    // Output layer parameters
    val weightHq: NDArray = normal(manager, Shape(numHiddens.toLong(), vocabSize.toLong()), device)
    val biasQ: NDArray = manager.zeros(Shape(vocabSize.toLong()), DataType.FLOAT32, device)

    // Attach gradients
    val params = NDList(weightXh, weightHh, biasH, weightHq, biasQ)
    for (param in params) {
        param.setRequiresGradient(true)
    }
    return params
}

/**
 * Executes normal.
 */
fun normal(
    manager: NDManager,
    shape: Shape,
    device: Device,
): NDArray = manager.randomNormal(0f, 0.01f, shape, DataType.FLOAT32, device)

/**
 * Executes initRNNState.
 */
fun initRNNState(
    manager: NDManager,
    batchSize: Int,
    numHiddens: Int,
    device: Device,
): NDList = NDList(manager.zeros(Shape(batchSize.toLong(), numHiddens.toLong()), DataType.FLOAT32, device))

/**
 * Executes rnn.
 */
fun rnn(
    inputs: NDArray,
    state: NDList,
    params: NDList,
): Pair<NDArray, NDList> {
    // Shape of `inputs`: (`numSteps`, `batchSize`, `vocabSize`)
    val weightXh = params[0]
    val weightHh = params[1]
    val biasH = params[2]
    val weightHq = params[3]
    val biasQ = params[4]
    var hiddenState = state[0]
    val outputs = NDList()
    // Shape of `X`: (`batchSize`, `vocabSize`)
    var inputStep: NDArray
    var outputStep: NDArray
    for (i in 0 until inputs.size(0)) {
        inputStep = inputs[i]
        hiddenState =
            inputStep
                .dot(weightXh)
                .add(hiddenState.dot(weightHh))
                .add(biasH)
                .tanh()
        outputStep = hiddenState.dot(weightHq).add(biasQ)
        outputs.add(outputStep)
    }
    return Pair(if (outputs.size > 1) NDArrays.concat(outputs) else outputs[0], NDList(hiddenState))
}

/**
 * Executes predictCh8.
 */
fun predictCh8(
    manager: NDManager,
    prefix: String,
    numPreds: Int,
    net: RNNModelScratch,
    vocab: Vocab,
    device: Device,
): String {
    var state: NDList = net.beginState(1, device)
    val outputs: MutableList<Int> = ArrayList()
    outputs.add(vocab.getIdx("" + prefix[0]))
    val getInput = {
        manager
            .create(outputs[outputs.size - 1])
            .toDevice(device, false)
            .reshape(Shape(1, 1))
    }
    for (c in prefix.substring(1).toCharArray()) { // Warm-up period
        state = net.forward(getInput(), state).second
        outputs.add(vocab.getIdx("" + c))
    }
    var y: NDArray
    for (i in 0 until numPreds) {
        val pair = net.forward(getInput(), state)
        y = pair.first
        state = pair.second
        outputs.add(
            y
                .argMax(1)
                .reshape(Shape(1))
                .getLong(0L)
                .toInt(),
        )
    }
    val output = StringBuilder()
    for (i in outputs) {
        output.append(vocab.idxToToken[i])
    }
    return output.toString()
}

/**
 * Executes gradClipping.
 */
fun gradClipping(
    net: RNNModelScratch,
    theta: Int,
    manager: NDManager,
) {
    var result = 0.0
    for (p in net.params) {
        val gradient = p.gradient
        gradient.attach(manager)
        result +=
            gradient
                .pow(2)
                .sum()
                .getFloat()
                .toDouble()
    }
    val norm = Math.sqrt(result)
    if (norm > theta) {
        for (param in net.params) {
            val gradient = param.gradient
            gradient.muli(theta / norm)
        }
    }
}

/**
 * Executes trainEpochCh8.
 */
fun trainEpochCh8(
    manager: NDManager,
    net: RNNModelScratch,
    trainIter: List<NDList>,
    loss: Loss,
    updater: (Int, NDManager) -> Unit,
    device: Device,
    useRandomIter: Boolean,
): Pair<Double, Double> {
    val watch = StopWatch()
    watch.start()
    val metric = Accumulator(2) // Sum of training loss, no. of tokens
    manager.newSubManager().use { childManager ->
        var state: NDList? = null
        for (pair in trainIter) {
            var features = pair[0].toDevice(device, true)
            features.attach(childManager)
            val labels = pair[1].toDevice(device, true)
            labels.attach(childManager)
            if (state == null || useRandomIter) {
                // Initialize `state` when either it is the first iteration or
                // using random sampling
                state = net.beginState(features.shape.shape[0].toInt(), device)
            } else {
                for (s in state) {
                    s.stopGradient()
                }
            }
            state.attach(childManager)
            var y = labels.transpose().reshape(Shape(-1))
            features = features.toDevice(device, false)
            y = y.toDevice(device, false)
            manager.engine.newGradientCollector().use { gc ->
                val pairResult = net.forward(features, state)
                val yHat: NDArray = pairResult.first
                state = pairResult.second
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
 * Executes trainCh8.
 */
fun trainCh8(
    manager: NDManager,
    net: RNNModelScratch,
    trainIter: List<NDList>,
    vocab: Vocab,
    lr: Int,
    numEpochs: Int,
    device: Device,
    useRandomIter: Boolean,
) {
    val loss = SoftmaxCrossEntropyLoss()
//    val animator = Animator()
    // Initialize
    val updater = { batchSize: Int, subManager: NDManager ->
        sgd(net.params, lr.toFloat(), batchSize, subManager)
    }
    val predict = { prefix: String -> predictCh8(manager, prefix, 50, net, vocab, device) }
    // Train and predict
    var ppl = 0.0
    var speed = 0.0
    for (epoch in 0 until numEpochs) {
        val pair = trainEpochCh8(manager, net, trainIter, loss, updater, device, useRandomIter)
        ppl = pair.first
        speed = pair.second
        if ((epoch + 1) % 10 == 0) {
//            animator.add(epoch + 1, ppl.toFloat(), "")
//            animator.show()
            println("${epoch + 1} : $ppl")
        }
    }
    println("perplexity: %.1f, %.1f tokens/sec on %s%n".format(ppl, speed, device.toString()))
    println(predict("time traveller"))
    println(predict("traveller"))
}

/**
 * Represents RnnNet.
 */
class RnnNet
