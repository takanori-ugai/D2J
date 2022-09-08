package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager
import jp.live.ugai.d2j.timemachine.TimeMachine.trainCh8
import jp.live.ugai.d2j.timemachine.TimeMachineDataset

fun main() {
    val manager = NDManager.newBaseManager()
    // Load data
    // Load data
    val batchSize = 32
    val numSteps = 35
    val device = manager.device
    val dataset = TimeMachineDataset.Builder()
        .setManager(manager)
        .setMaxTokens(10000)
        .setSampling(batchSize, false)
        .setSteps(numSteps)
        .build()
    dataset.prepare()
    val vocab = dataset.vocab

// Define the bidirectional LSTM model by setting `bidirectional=True`

// Define the bidirectional LSTM model by setting `bidirectional=True`
    val vocabSize = vocab!!.length()
    val numHiddens = 256
    val numLayers = 2
    val lstmLayer = LSTM0.builder()
        .setNumLayers(numLayers)
        .setStateSize(numHiddens)
        .optReturnState(true)
        .optBatchFirst(false)
        .optBidirectional(true)
        .build()

// Train the model

// Train the model
    val model = RNNModel(lstmLayer, vocabSize)
    val numEpochs = Integer.getInteger("MAX_EPOCH", 500)

    val lr = 1
    trainCh8(model, dataset, vocab, lr, numEpochs, device, false, manager)
}

class BiRnn
