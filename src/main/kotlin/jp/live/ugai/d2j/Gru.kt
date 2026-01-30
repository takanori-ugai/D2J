package jp.live.ugai.d2j

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.recurrent.GRU
import jp.live.ugai.d2j.timemachine.TimeMachine.trainCh8
import jp.live.ugai.d2j.timemachine.TimeMachineDataset
import jp.live.ugai.d2j.util.NDArrayUtils

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

    fun getParams(
        vocabSize: Int,
        numHiddens: Int,
        device: Device,
    ): NDList {
        // Update gate parameters
        var temp = NDArrayUtils.three(manager, vocabSize, numHiddens, device)
        val weightXz = temp[0]
        val weightHz = temp[1]
        val biasZ = temp[2]

        // Reset gate parameters
        temp = NDArrayUtils.three(manager, vocabSize, numHiddens, device)
        val weightXr = temp[0]
        val weightHr = temp[1]
        val biasR = temp[2]

        // Candidate hidden state parameters
        temp = NDArrayUtils.three(manager, vocabSize, numHiddens, device)
        val weightXh = temp[0]
        val weightHh = temp[1]
        val biasH = temp[2]

        // Output layer parameters
        val weightHq =
            NDArrayUtils.normal(manager, Shape(numHiddens.toLong(), vocabSize.toLong()), device)
        val biasQ: NDArray = manager.zeros(Shape(vocabSize.toLong()), DataType.FLOAT32, device)

        // Attach gradients
        val params =
            NDList(
                weightXz,
                weightHz,
                biasZ,
                weightXr,
                weightHr,
                biasR,
                weightXh,
                weightHh,
                biasH,
                weightHq,
                biasQ,
            )
        for (param in params) {
            param.setRequiresGradient(true)
        }
        return params
    }

    fun initGruState(
        batchSize: Int,
        numHiddens: Int,
        device: Device,
    ): NDList = NDList(manager.zeros(Shape(batchSize.toLong(), numHiddens.toLong()), DataType.FLOAT32, device))

    fun gru(
        inputs: NDArray,
        state: NDList,
        params: NDList,
    ): Pair<NDArray, NDList> {
        val weightXz = params[0]
        val weightHz = params[1]
        val biasZ = params[2]
        val weightXr = params[3]
        val weightHr = params[4]
        val biasR = params[5]
        val weightXh = params[6]
        val weightHh = params[7]
        val biasH = params[8]
        val weightHq = params[9]
        val biasQ = params[10]
        var hiddenState = state[0]
        val outputs = NDList()
        var inputStep: NDArray
        var outputStep: NDArray
        var updateGate: NDArray
        var resetGate: NDArray
        var candidateState: NDArray
        for (i in 0 until inputs.size(0)) {
            inputStep = inputs[i]
            updateGate = Activation.sigmoid(inputStep.dot(weightXz).add(hiddenState.dot(weightHz).add(biasZ)))
            resetGate = Activation.sigmoid(inputStep.dot(weightXr).add(hiddenState.dot(weightHr).add(biasR)))
            candidateState =
                Activation.tanh(inputStep.dot(weightXh).add(resetGate.mul(hiddenState).dot(weightHh).add(biasH)))
            hiddenState = updateGate.mul(hiddenState).add(updateGate.mul(-1).add(1).mul(candidateState))
            outputStep = hiddenState.dot(weightHq).add(biasQ)
            outputs.add(outputStep)
        }
        return Pair(
            if (outputs.size > 1) NDArrays.concat(outputs) else outputs[0],
            NDList(hiddenState),
        )
    }

    val vocabSize = vocab!!.length()
    val numHiddens = 256
    val device = manager.device
    val numEpochs = Integer.getInteger("MAX_EPOCH", 500)

    val lr = 1

    val gruLayer =
        GRU
            .builder()
            .setNumLayers(1)
            .setStateSize(numHiddens)
            .optReturnState(true)
            .optBatchFirst(false)
            .build()
    val modelConcise = RNNModel(gruLayer, vocab.length())
    trainCh8(modelConcise, dataset, vocab, lr, numEpochs, device, false, manager)
}

/**
 * Placeholder for a dedicated GRU example container.
 */
internal class Gru
