package jp.live.ugai.d2j

import ai.djl.Device
import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.nn.recurrent.RNN
import ai.djl.nn.recurrent.RNN.Activation
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.Trainer
import ai.djl.training.dataset.RandomAccessDataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import jp.live.ugai.d2j.timemachine.RNNModelScratch
import jp.live.ugai.d2j.timemachine.TimeMachineDataset
import jp.live.ugai.d2j.timemachine.Vocab
import jp.live.ugai.d2j.util.Accumulator
import jp.live.ugai.d2j.util.StopWatch
import jp.live.ugai.d2j.util.Training.sgd

/**
 * Constant BATCH_SIZE.
 */
const val BATCH_SIZE = 32

/**
 * Constant NUM_STEPS.
 */
const val NUM_STEPS = 35

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
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.training.listener", "WARN")

    val manager = NDManager.newBaseManager()

    val dataset: TimeMachineDataset =
        TimeMachineDataset
            .Builder()
            .setManager(manager)
            .setMaxTokens(10000)
            .setSampling(BATCH_SIZE, false)
            .setSteps(NUM_STEPS)
            .build()
    dataset.prepare()
    val vocab = dataset.vocab
    val numHiddens = 256
    val rnnLayer =
        RNN
            .builder()
            .setNumLayers(1)
            .setStateSize(numHiddens)
            .setActivation(Activation.RELU)
            .optReturnState(true)
            .optBatchFirst(false)
            .build()
    val state = beginState(BATCH_SIZE, 1, numHiddens, manager)
    println(state.size)
    println(state[0].shape)

    val inputTensor =
        manager.randomUniform(
            0.0f,
            1.0f,
            Shape(NUM_STEPS.toLong(), BATCH_SIZE.toLong(), vocab!!.length().toLong()),
        )

    val input = NDList(inputTensor, state[0])
    rnnLayer.initialize(manager, DataType.FLOAT32, input.shapes[0], input.shapes[1])
    flattenParametersIfAvailable(rnnLayer)
    val forwardOutput = rnnLayer.forward(ParameterStore(manager, false), input, false)
    val rnnOutput = forwardOutput[0]
    val stateNew = forwardOutput[1]

    println(rnnOutput.shape)
    println(stateNew.shape)

    val device = manager.device
    val net = RNNModel(rnnLayer, vocab.length())
    net.initialize(manager, DataType.FLOAT32, inputTensor.shape)
    println(predictCh8("time traveller", 10, net, vocab, device, manager))

    val numEpochs: Int = Integer.getInteger("MAX_EPOCH", 500)
    val lr = 1.0f
    trainCh8(net, dataset, vocab, lr, numEpochs, device, false, manager)
    predictCh8("time traveller", 10, net, vocab, device, manager)
}

/**
 * Executes beginState.
 */
fun beginState(
    batchSize: Int,
    numLayers: Int,
    numHiddens: Int,
    manager: NDManager,
): NDList = NDList(manager.zeros(Shape(numLayers.toLong(), batchSize.toLong(), numHiddens.toLong())))

/**
 * Executes predictCh8.
 */
fun predictCh8(
    prefix: String,
    numPreds: Int,
    net: Any,
    vocab: Vocab,
    device: Device,
    manager: NDManager,
): String {
    val outputs: MutableList<Int> = ArrayList()
    outputs.add(vocab.getIdx("" + prefix[0]))
    val getInput = {
        manager
            .create(outputs[outputs.size - 1])
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
        var y: NDArray
        for (i in 0 until numPreds) {
            val pair = castedNet.forward(getInput(), state)
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
    } else {
        val castedNet = net as AbstractBlock
        var state: NDList? = null
        for (c in prefix.substring(1).toCharArray()) { // Warm-up period
            state =
                if (state == null) {
                    // Begin state
                    castedNet
                        .forward(
                            ParameterStore(manager, false),
                            NDList(getInput()),
                            false,
                        ).subNDList(1)
                } else {
                    castedNet
                        .forward(
                            ParameterStore(manager, false),
                            NDList(getInput()).addAll(state),
                            false,
                        ).subNDList(1)
                }
            outputs.add(vocab.getIdx("" + c))
        }
        var y: NDArray
        for (i in 0 until numPreds) {
            val pair =
                castedNet.forward(
                    ParameterStore(manager, false),
                    NDList(getInput()).addAll(state),
                    false,
                )
            y = pair[0]
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
    val output = StringBuilder()
    for (i in outputs) {
        output.append(vocab.idxToToken[i])
    }
    return output.toString()
}

/**
 * Executes trainCh8.
 */
fun trainCh8(
    net: Any,
    dataset: RandomAccessDataset,
    vocab: Vocab,
    lr: Float,
    numEpochs: Int,
    device: Device,
    useRandomIter: Boolean,
    manager: NDManager,
) {
    val loss = SoftmaxCrossEntropyLoss()
    var model: Model? = null
    var trainer: Trainer? = null
//    val animator = Animator()
    val updater: (Int, NDManager) -> Unit =
        if (net is RNNModelScratch) {
            { batchSize: Int, subManager: NDManager ->
                sgd(net.params, lr, batchSize, subManager)
            }
        } else {
            // Already initialized net
            val castedNet = net as AbstractBlock
            val lrt: Tracker = Tracker.fixed(lr)
            val sgd: Optimizer = Optimizer.sgd().setLearningRateTracker(lrt).build()
            val config: DefaultTrainingConfig =
                DefaultTrainingConfig(loss)
                    .optOptimizer(sgd) // Optimizer (loss function)
                    .optInitializer(NormalInitializer(0.01f), Parameter.Type.WEIGHT) // setting the initializer
                    .optDevices(Engine.getInstance().getDevices(1)) // setting the number of GPUs needed
                    .addEvaluator(Accuracy()) // Model Accuracy
                    .also { cfg ->
                        TrainingListener.Defaults.logging().forEach { cfg.addTrainingListeners(it) }
                    } // Logging
            model = Model.newInstance("model").also { it.block = castedNet }
            trainer = model!!.newTrainer(config)
            { _: Int, _: NDManager ->
                trainer!!.step()
            }
        }
    val predict: (String) -> String =
        { prefix ->
            predictCh8(prefix, 50, net, vocab, device, manager)
        }
    // Train and predict
    var ppl = 0.0
    var speed = 0.0
    try {
        for (epoch in 0 until numEpochs) {
            val pair = trainEpochCh8(net, dataset, loss, updater, device, useRandomIter, manager)
            ppl = pair.first
            speed = pair.second
            if ((epoch + 1) % 10 == 0) {
//                animator.add(epoch + 1, ppl.toFloat(), "")
//                animator.show()
                println("${epoch + 1} : $ppl")
            }
        }
    } finally {
        trainer?.close()
        model?.close()
    }
    println(
        "perplexity: %.1f, %.1f tokens/sec on %s%n".format(ppl, speed, device.toString()),
    )
    println(predict("time traveller"))
    println(predict("traveller"))
}

/**
 * Executes trainEpochCh8.
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
            var inputBatch = batch.data.head().toDevice(device, true)
            val labelBatch = batch.labels.head().toDevice(device, true)
            if (state == null || useRandomIter) {
                // Initialize `state` when either it is the first iteration or
                // using random sampling
                if (net is RNNModelScratch) {
                    state = net.beginState(inputBatch.shape.shape[0].toInt(), device)
                }
            } else {
                val detachedState = NDList()
                for (s in state) {
                    s.stopGradient()
                    detachedState.add(s.duplicate())
                }
                state = detachedState
            }
            state?.attach(childManager)
            var labelFlat = labelBatch.transpose().reshape(Shape(-1))
            inputBatch = inputBatch.toDevice(device, false)
            labelFlat = labelFlat.toDevice(device, false)
            Engine.getInstance().newGradientCollector().use { gc ->
                val yHat: NDArray
                if (net is RNNModelScratch) {
                    val pairResult = net.forward(inputBatch, state!!)
                    yHat = pairResult.first
                    state = pairResult.second
                } else {
                    val pairResult: NDList
                    pairResult =
                        if (state == null) {
                            // Begin state
                            (net as AbstractBlock)
                                .forward(
                                    ParameterStore(childManager, false),
                                    NDList(inputBatch),
                                    true,
                                )
                        } else {
                            (net as AbstractBlock)
                                .forward(
                                    ParameterStore(childManager, false),
                                    NDList(inputBatch).addAll(state),
                                    true,
                                )
                        }
                    yHat = pairResult[0]
                    state = pairResult.subNDList(1)
                }
                val l = loss.evaluate(NDList(labelFlat), NDList(yHat)).mean()
                gc.backward(l)
                metric.add(floatArrayOf(l.getFloat() * labelFlat.size(), labelFlat.size().toFloat()))
            }
            gradClipping(net, 1, childManager)
            updater(1, childManager) // Since the `mean` function has been invoked
        }
    }
    return Pair(Math.exp((metric.get(0) / metric.get(1)).toDouble()), metric.get(1) / watch.stop())
}

/**
 * Executes gradClipping.
 */
fun gradClipping(
    net: Any,
    theta: Int,
    manager: NDManager,
) {
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
        result += gradient.pow(2).sum().getFloat()
    }
    val norm = Math.sqrt(result)
    if (norm > theta) {
        for (param in params) {
            val gradient = param.gradient
            gradient.muli(theta / norm)
        }
    }
}
