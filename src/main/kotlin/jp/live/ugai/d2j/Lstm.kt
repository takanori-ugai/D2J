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

fun main() {
    val manager = NDManager.newBaseManager()
    val batchSize = 32
    val numSteps = 35

    val dataset =
        TimeMachineDataset.Builder()
            .setManager(manager)
            .setMaxTokens(10000)
            .setSampling(batchSize, false)
            .setSteps(numSteps)
            .build()
    dataset.prepare()
    val vocab = dataset.vocab

    fun normal(
        shape: Shape,
        device: Device,
    ): NDArray {
        return manager.randomNormal(0.0f, 0.01f, shape, DataType.FLOAT32, device)
    }

    fun three(
        numInputs: Int,
        numHiddens: Int,
        device: Device,
    ): NDList {
        return NDList(
            normal(Shape(numInputs.toLong(), numHiddens.toLong()), device),
            normal(Shape(numHiddens.toLong(), numHiddens.toLong()), device),
            manager.zeros(Shape(numHiddens.toLong()), DataType.FLOAT32, device),
        )
    }

    fun getLSTMParams(
        vocabSize: Int,
        numHiddens: Int,
        device: Device,
    ): NDList {
        // Input gate parameters
        var temp: NDList = three(vocabSize, numHiddens, device)
        val W_xi: NDArray = temp.get(0)
        val W_hi: NDArray = temp.get(1)
        val b_i: NDArray = temp.get(2)

        // Forget gate parameters
        temp = three(vocabSize, numHiddens, device)
        val W_xf: NDArray = temp.get(0)
        val W_hf: NDArray = temp.get(1)
        val b_f: NDArray = temp.get(2)

        // Output gate parameters
        temp = three(vocabSize, numHiddens, device)
        val W_xo: NDArray = temp.get(0)
        val W_ho: NDArray = temp.get(1)
        val b_o: NDArray = temp.get(2)

        // Candidate memory cell parameters
        temp = three(vocabSize, numHiddens, device)
        val W_xc: NDArray = temp.get(0)
        val W_hc: NDArray = temp.get(1)
        val b_c: NDArray = temp.get(2)

        // Output layer parameters
        val W_hq: NDArray = normal(Shape(numHiddens.toLong(), vocabSize.toLong()), device)
        val b_q: NDArray = manager.zeros(Shape(vocabSize.toLong()), DataType.FLOAT32, device)

        // Attach gradients
        val params = NDList(W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q)
        for (param in params) {
            param.setRequiresGradient(true)
        }
        return params
    }

    fun initLSTMState(
        batchSize: Int,
        numHiddens: Int,
        device: Device,
    ): NDList {
        return NDList(
            manager.zeros(Shape(batchSize.toLong(), numHiddens.toLong()), DataType.FLOAT32, device),
            manager.zeros(Shape(batchSize.toLong(), numHiddens.toLong()), DataType.FLOAT32, device),
        )
    }

    fun lstm(
        inputs: NDArray,
        state: NDList,
        params: NDList,
    ): Pair<NDArray, NDList> {
        val W_xi = params[0]
        val W_hi = params[1]
        val b_i = params[2]
        val W_xf = params[3]
        val W_hf = params[4]
        val b_f = params[5]
        val W_xo = params[6]
        val W_ho = params[7]
        val b_o = params[8]
        val W_xc = params[9]
        val W_hc = params[10]
        val b_c = params[11]
        val W_hq = params[12]
        val b_q = params[13]
        var H = state[0]
        var C = state[1]
        val outputs = NDList()
        var X: NDArray
        var Y: NDArray
        var I: NDArray
        var F: NDArray
        var O: NDArray
        var C_tilda: NDArray
        for (i in 0 until inputs.size(0)) {
            X = inputs[i]
            I = Activation.sigmoid(X.dot(W_xi).add(H.dot(W_hi).add(b_i)))
            F = Activation.sigmoid(X.dot(W_xf).add(H.dot(W_hf).add(b_f)))
            O = Activation.sigmoid(X.dot(W_xo).add(H.dot(W_ho).add(b_o)))
            C_tilda = Activation.tanh(X.dot(W_xc).add(H.dot(W_hc).add(b_c)))
            C = F.mul(C).add(I.mul(C_tilda))
            H = O.mul(Activation.tanh(C))
            Y = H.dot(W_hq).add(b_q)
            outputs.add(Y)
        }
        return Pair(if (outputs.size > 1) NDArrays.concat(outputs) else outputs[0], NDList(H, C))
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
        LSTM.builder()
            .setNumLayers(1)
            .setStateSize(numHiddens)
            .optReturnState(true)
            .optBatchFirst(false)
            .build()
    val modelConcise = RNNModel(lstmLayer, vocab.length())
    trainCh8(modelConcise, dataset, vocab, lr, numEpochs, device, false, manager)
}

class Lstm
