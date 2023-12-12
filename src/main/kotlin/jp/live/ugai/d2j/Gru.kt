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

    fun getParams(
        vocabSize: Int,
        numHiddens: Int,
        device: Device,
    ): NDList {
        // Update gate parameters
        var temp = three(vocabSize, numHiddens, device)
        val W_xz = temp[0]
        val W_hz = temp[1]
        val b_z = temp[2]

        // Reset gate parameters
        temp = three(vocabSize, numHiddens, device)
        val W_xr = temp[0]
        val W_hr = temp[1]
        val b_r = temp[2]

        // Candidate hidden state parameters
        temp = three(vocabSize, numHiddens, device)
        val W_xh = temp[0]
        val W_hh = temp[1]
        val b_h = temp[2]

        // Output layer parameters
        val W_hq = normal(Shape(numHiddens.toLong(), vocabSize.toLong()), device)
        val b_q: NDArray = manager.zeros(Shape(vocabSize.toLong()), DataType.FLOAT32, device)

        // Attach gradients
        val params = NDList(W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q)
        for (param in params) {
            param.setRequiresGradient(true)
        }
        return params
    }

    fun initGruState(
        batchSize: Int,
        numHiddens: Int,
        device: Device,
    ): NDList {
        return NDList(manager.zeros(Shape(batchSize.toLong(), numHiddens.toLong()), DataType.FLOAT32, device))
    }

    fun gru(
        inputs: NDArray,
        state: NDList,
        params: NDList,
    ): Pair<NDArray, NDList> {
        val W_xz = params[0]
        val W_hz = params[1]
        val b_z = params[2]
        val W_xr = params[3]
        val W_hr = params[4]
        val b_r = params[5]
        val W_xh = params[6]
        val W_hh = params[7]
        val b_h = params[8]
        val W_hq = params[9]
        val b_q = params[10]
        var H = state[0]
        val outputs = NDList()
        var X: NDArray
        var Y: NDArray
        var Z: NDArray
        var R: NDArray
        var H_tilda: NDArray
        for (i in 0 until inputs.size(0)) {
            X = inputs[i]
            Z = Activation.sigmoid(X.dot(W_xz).add(H.dot(W_hz).add(b_z)))
            R = Activation.sigmoid(X.dot(W_xr).add(H.dot(W_hr).add(b_r)))
            H_tilda = Activation.tanh(X.dot(W_xh).add(R.mul(H).dot(W_hh).add(b_h)))
            H = Z.mul(H).add(Z.mul(-1).add(1).mul(H_tilda))
            Y = H.dot(W_hq).add(b_q)
            outputs.add(Y)
        }
        return Pair(if (outputs.size > 1) NDArrays.concat(outputs) else outputs[0], NDList(H))
    }

    val vocabSize = vocab!!.length()
    val numHiddens = 256
    val device = manager.device
    val numEpochs = Integer.getInteger("MAX_EPOCH", 500)

    val lr = 1

    val getParamsFn = ::getParams
    val initGruStateFn = ::initGruState
    val gruFn = ::gru

//    val model = RNNModelScratch(vocabSize, numHiddens, device, getParamsFn, initGruStateFn, gruFn)
//    trainCh8(model, dataset, vocab, lr, numEpochs, device, false, manager)

    val gruLayer =
        GRU.builder()
            .setNumLayers(1)
            .setStateSize(numHiddens)
            .optReturnState(true)
            .optBatchFirst(false)
            .build()
    val modelConcise = RNNModel(gruLayer, vocab.length())
    trainCh8(modelConcise, dataset, vocab, lr, numEpochs, device, false, manager)
}

class Gru
