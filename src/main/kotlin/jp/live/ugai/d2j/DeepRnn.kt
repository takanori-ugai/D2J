package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager
import jp.live.ugai.d2j.timemachine.TimeMachine.trainCh8
import jp.live.ugai.d2j.timemachine.TimeMachineDataset

/**
 * Executes main.
 */
fun main() {
    val manager = NDManager.newBaseManager()
    val batchSize = 32
    val numSteps = 35

    val dataset =
        TimeMachineDataset
            .Builder()
            .setManager(manager)
            .setMaxTokens(10000)
            .setSampling(batchSize, false)
            .setSteps(numSteps)
            .build()
    dataset.prepare()
    val vocab = dataset.vocab

    val vocabSize = vocab!!.length()
    val numHiddens = 256
    val numLayers = 1
    val device = manager.device
    val lstmLayer =
        LSTM0
            .builder()
            .setNumLayers(numLayers)
            .setStateSize(numHiddens)
            .optReturnState(true)
            .optBatchFirst(false)
            .build()

    val model = RNNModel(lstmLayer, vocabSize)

    val numEpochs = Integer.getInteger("MAX_EPOCH", 500)

    val lr = 2
    trainCh8(model, dataset, vocab, lr, numEpochs, device, false, manager)
}

/**
 * Represents DeepRnn.
 */
class DeepRnn
