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

val manager = NDManager.newBaseManager()

fun main() {
    val batchSize = 32
    val numSteps = 35
    val timeMachine: Pair<List<NDList>, Vocab> =
        SeqDataLoader.loadDataTimeMachine(batchSize, numSteps, false, 10000)
    val trainIter: List<NDList> = timeMachine.first
    val vocab: Vocab = timeMachine.second

    println(vocab.length())
    println(manager.create(intArrayOf(0, 2)).oneHot(vocab.length()))
    var X = manager.arange(10).reshape(Shape(2, 5))
    println(X.transpose().oneHot(28).shape)

    val numHiddens = 512
    val getParamsFn = ::getParams
    val initRNNStateFn = ::initRNNState
    val rnnFn = ::rnn

    X = manager.arange(10).reshape(Shape(2, 5))
    val device = manager.device
    val net = RNNModelScratch(vocab.length(), numHiddens, device, getParamsFn, initRNNStateFn, rnnFn)
    val state = net.beginState(X.shape.shape[0].toInt(), device)
    val pairResult: Pair<NDArray, NDList> = net.forward(X.toDevice(device, false), state)
    val Y: NDArray = pairResult.first
    val newState: NDList = pairResult.second
    println(Y.shape)
    println(newState[0].shape)

    println(predictCh8("time traveller ", 10, net, vocab, manager.device))

    val numEpochs = Integer.getInteger("MAX_EPOCH", 500)

    val lr = 1
    trainCh8(net, trainIter, vocab, lr, numEpochs, manager.device, false)
}

fun getParams(vocabSize: Int, numHiddens: Int, device: Device): NDList {
    // Hidden layer parameters
    val W_xh: NDArray = normal(Shape(vocabSize.toLong(), numHiddens.toLong()), device)
    val W_hh: NDArray = normal(Shape(numHiddens.toLong(), numHiddens.toLong()), device)
    val b_h: NDArray = manager.zeros(Shape(numHiddens.toLong()), DataType.FLOAT32, device)
    // Output layer parameters
    val W_hq: NDArray = normal(Shape(numHiddens.toLong(), vocabSize.toLong()), device)
    val b_q: NDArray = manager.zeros(Shape(vocabSize.toLong()), DataType.FLOAT32, device)

    // Attach gradients
    val params = NDList(W_xh, W_hh, b_h, W_hq, b_q)
    for (param in params) {
        param.setRequiresGradient(true)
    }
    return params
}

fun normal(shape: Shape, device: Device): NDArray {
    return manager.randomNormal(0f, 0.01f, shape, DataType.FLOAT32, device)
}

fun initRNNState(batchSize: Int, numHiddens: Int, device: Device): NDList {
    return NDList(manager.zeros(Shape(batchSize.toLong(), numHiddens.toLong()), DataType.FLOAT32, device))
}

fun rnn(inputs: NDArray, state: NDList, params: NDList): Pair<NDArray, NDList> {
    // Shape of `inputs`: (`numSteps`, `batchSize`, `vocabSize`)
    val W_xh = params[0]
    val W_hh = params[1]
    val b_h = params[2]
    val W_hq = params[3]
    val b_q = params[4]
    var H = state[0]
    val outputs = NDList()
    // Shape of `X`: (`batchSize`, `vocabSize`)
    var X: NDArray
    var Y: NDArray
    for (i in 0 until inputs.size(0)) {
        X = inputs[i]
        H = X.dot(W_xh).add(H.dot(W_hh)).add(b_h).tanh()
        Y = H.dot(W_hq).add(b_q)
        outputs.add(Y)
    }
    return Pair(if (outputs.size > 1) NDArrays.concat(outputs) else outputs[0], NDList(H))
}

fun predictCh8(prefix: String, numPreds: Int, net: RNNModelScratch, vocab: Vocab, device: Device): String {
    var state: NDList = net.beginState(1, device)
    val outputs: MutableList<Int> = ArrayList()
    outputs.add(vocab.getIdx("" + prefix[0]))
    val getInput = {
        manager.create(outputs[outputs.size - 1])
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
        outputs.add(y.argMax(1).reshape(Shape(1)).getLong(0L).toInt())
    }
    val output = StringBuilder()
    for (i in outputs) {
        output.append(vocab.idxToToken[i])
    }
    return output.toString()
}

/** Clip the gradient.  */
fun gradClipping(net: RNNModelScratch, theta: Int, manager: NDManager) {
    var result = 0.0
    for (p in net.params) {
        val gradient = p.gradient
        gradient.attach(manager)
        result += gradient.pow(2).sum().getFloat().toDouble()
    }
    val norm = Math.sqrt(result)
    if (norm > theta) {
        for (param in net.params) {
            val gradient = param.gradient
            gradient.muli(theta / norm)
        }
    }
}

/** Train a model within one epoch.  */
fun trainEpochCh8(
    net: RNNModelScratch,
    trainIter: List<NDList>,
    loss: Loss,
    updater: (Int, NDManager) -> Unit,
    device: Device,
    useRandomIter: Boolean
): Pair<Double, Double> {
    val watch = StopWatch()
    watch.start()
    val metric = Accumulator(2) // Sum of training loss, no. of tokens
    manager.newSubManager().use { childManager ->
        var state: NDList? = null
        for (pair in trainIter) {
            var X = pair[0].toDevice(device, true)
            X.attach(childManager)
            val Y = pair[1].toDevice(device, true)
            Y.attach(childManager)
            if (state == null || useRandomIter) {
                // Initialize `state` when either it is the first iteration or
                // using random sampling
                state = net.beginState(X.shape.shape[0].toInt(), device)
            } else {
                for (s in state) {
                    s.stopGradient()
                }
            }
            state.attach(childManager)
            var y = Y.transpose().reshape(Shape(-1))
            X = X.toDevice(device, false)
            y = y.toDevice(device, false)
            manager.engine.newGradientCollector().use { gc ->
                val pairResult = net.forward(X, state!!)
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

/** Train a model.  */
fun trainCh8(
    net: RNNModelScratch,
    trainIter: List<NDList>,
    vocab: Vocab,
    lr: Int,
    numEpochs: Int,
    device: Device,
    useRandomIter: Boolean
) {
    val loss = SoftmaxCrossEntropyLoss()
//    val animator = Animator()
    // Initialize
    val updater = { batchSize: Int, subManager: NDManager ->
        sgd(net.params, lr.toFloat(), batchSize, subManager)
    }
    val predict = { prefix: String -> predictCh8(prefix, 50, net, vocab, device) }
    // Train and predict
    var ppl = 0.0
    var speed = 0.0
    for (epoch in 0 until numEpochs) {
        val pair = trainEpochCh8(net, trainIter, loss, updater, device, useRandomIter)
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

class RnnNet
