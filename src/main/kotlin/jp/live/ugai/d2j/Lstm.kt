package jp.live.ugai.d2j

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.recurrent.LSTM
import jp.live.ugai.d2j.timemachine.RNNModelScratch
import jp.live.ugai.d2j.timemachine.TimeMachine.trainCh8
import jp.live.ugai.d2j.timemachine.TimeMachineDataset
import jp.live.ugai.d2j.util.NDArrayUtils

/**
 * Demonstrates training LSTM models on the TimeMachine dataset.
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

    fun getLSTMParams(
        vocabSize: Int,
        numHiddens: Int,
        device: Device,
    ): NDList {
        // Input gate parameters
        var temp: NDList = NDArrayUtils.three(manager, vocabSize, numHiddens, device)
        val weightXi: NDArray = temp.get(0)
        val weightHi: NDArray = temp.get(1)
        val biasI: NDArray = temp.get(2)

        // Forget gate parameters
        temp = NDArrayUtils.three(manager, vocabSize, numHiddens, device)
        val weightXf: NDArray = temp.get(0)
        val weightHf: NDArray = temp.get(1)
        val biasF: NDArray = temp.get(2)

        // Output gate parameters
        temp = NDArrayUtils.three(manager, vocabSize, numHiddens, device)
        val weightXo: NDArray = temp.get(0)
        val weightHo: NDArray = temp.get(1)
        val biasO: NDArray = temp.get(2)

        // Candidate memory cell parameters
        temp = NDArrayUtils.three(manager, vocabSize, numHiddens, device)
        val weightXc: NDArray = temp.get(0)
        val weightHc: NDArray = temp.get(1)
        val biasC: NDArray = temp.get(2)

        // Output layer parameters
        val weightHq: NDArray =
            NDArrayUtils.normal(manager, Shape(numHiddens.toLong(), vocabSize.toLong()), device)
        val biasQ: NDArray = manager.zeros(Shape(vocabSize.toLong()), DataType.FLOAT32, device)

        // Attach gradients
        val params =
            NDList(
                weightXi,
                weightHi,
                biasI,
                weightXf,
                weightHf,
                biasF,
                weightXo,
                weightHo,
                biasO,
                weightXc,
                weightHc,
                biasC,
                weightHq,
                biasQ,
            )
        for (param in params) {
            param.setRequiresGradient(true)
        }
        return params
    }

    fun initLSTMState(
        batchSize: Int,
        numHiddens: Int,
        device: Device,
    ): NDList =
        NDList(
            manager.zeros(Shape(batchSize.toLong(), numHiddens.toLong()), DataType.FLOAT32, device),
            manager.zeros(Shape(batchSize.toLong(), numHiddens.toLong()), DataType.FLOAT32, device),
        )

    fun lstm(
        inputs: NDArray,
        state: NDList,
        params: NDList,
    ): Pair<NDArray, NDList> {
        val weightXi = params[0]
        val weightHi = params[1]
        val biasI = params[2]
        val weightXf = params[3]
        val weightHf = params[4]
        val biasF = params[5]
        val weightXo = params[6]
        val weightHo = params[7]
        val biasO = params[8]
        val weightXc = params[9]
        val weightHc = params[10]
        val biasC = params[11]
        val weightHq = params[12]
        val biasQ = params[13]
        var hiddenState = state[0]
        var cellState = state[1]
        val outputs = NDList()
        var inputStep: NDArray
        var outputStep: NDArray
        var inputGate: NDArray
        var forgetGate: NDArray
        var outputGate: NDArray
        var candidateCell: NDArray
        for (i in 0 until inputs.size(0)) {
            inputStep = inputs[i]
            inputGate = Activation.sigmoid(inputStep.dot(weightXi).add(hiddenState.dot(weightHi).add(biasI)))
            forgetGate = Activation.sigmoid(inputStep.dot(weightXf).add(hiddenState.dot(weightHf).add(biasF)))
            outputGate = Activation.sigmoid(inputStep.dot(weightXo).add(hiddenState.dot(weightHo).add(biasO)))
            candidateCell = Activation.tanh(inputStep.dot(weightXc).add(hiddenState.dot(weightHc).add(biasC)))
            cellState = forgetGate.mul(cellState).add(inputGate.mul(candidateCell))
            hiddenState = outputGate.mul(Activation.tanh(cellState))
            outputStep = hiddenState.dot(weightHq).add(biasQ)
            outputs.add(outputStep)
        }
        return Pair(
            if (outputs.size > 1) NDArrays.concat(outputs) else outputs[0],
            NDList(hiddenState, cellState),
        )
    }

    val vocabSize = vocab!!.length()
    val numHiddens = 256
    val device = manager.device
    val numEpochs = Integer.getInteger("MAX_EPOCH", 500)

    val lr = 1

    val getParamsFn = ::getLSTMParams
    val initLSTMStateFn = ::initLSTMState
    val lstmFn = ::lstm
    val model = RNNModelScratch(vocabSize, numHiddens, device, getParamsFn, initLSTMStateFn, lstmFn)
    trainCh8(model, dataset, vocab, lr, numEpochs, device, false, manager)

    val lstmLayer =
        LSTM
            .builder()
            .setNumLayers(1)
            .setStateSize(numHiddens)
            .optReturnState(true)
            .optBatchFirst(false)
            .build()
    val modelConcise = RNNModel(lstmLayer, vocab.length())
    trainCh8(modelConcise, dataset, vocab, lr, numEpochs, device, false, manager)
}
