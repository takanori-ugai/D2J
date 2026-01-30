package jp.live.ugai.d2j

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.loss.Loss
import jp.live.ugai.d2j.lstm.EncoderDecoder
import jp.live.ugai.d2j.timemachine.Vocab
import jp.live.ugai.d2j.util.NMT
import jp.live.ugai.d2j.util.NMT.loadDataNMT
import java.util.Locale

/**
 * Demonstrates sequence-to-sequence training and inference on a toy dataset.
 */
fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    val manager = NDManager.newBaseManager()
    val ps = ParameterStore(manager, false)
    var encoder = Seq2SeqEncoder(10, 8, 16, 2, 0.0f)
    var sampleInput = manager.zeros(Shape(4, 7))
    encoder.initialize(manager, DataType.FLOAT32, sampleInput.shape)
    var outputState = encoder.forward(ps, NDList(sampleInput), false)
    var output = outputState.head()

    println(output.shape)

    var state = outputState.subNDList(1)
    println(state.size)
    println(state.head().shape)

    val decoder = Seq2SeqDecoder(10, 8, 16, 2, 0f)
    state = decoder.initState(outputState)
    val input = NDList(sampleInput).addAll(state)
    decoder.initialize(manager, DataType.FLOAT32, input.shapes[0], input.shapes[1])
    outputState = decoder.forward(ps, input, false)

    output = outputState.head()
    println(output.shape)

    state = outputState.subNDList(1)
    println(state.size)
    println(state.head().shape)

    val sampleMaskInput = manager.create(arrayOf(intArrayOf(1, 2, 3), intArrayOf(4, 5, 6)))
    println(sequenceMask(sampleMaskInput, manager.create(intArrayOf(1, 2))))

    val sampleMaskTensor = manager.ones(Shape(2, 3, 4))
    println(sequenceMask(sampleMaskTensor, manager.create(intArrayOf(1, 2)), -1f))

    val loss: Loss = MaskedSoftmaxCELoss()
    val labels =
        NDList(
            manager.create(
                arrayOf(
                    intArrayOf(1, 2, 3, 4),
                    intArrayOf(2, 3, 4, 0),
                    intArrayOf(3, 4, 0, 0),
                ),
            ),
        )
    labels.add(manager.create(intArrayOf(4, 2, 2)))
    val predictions = NDList(manager.ones(Shape(3, 4, 10)))
    println(loss.evaluate(labels, predictions))

    val embedSize = 32
    val numHiddens = 32
    val numLayers = 2
    val batchSize = 64
    val numSteps = 10
    val numEpochs = Integer.getInteger("MAX_EPOCH", 100)

    val dropout = 0.1f
    val lr = 0.005f
    val device = manager.device

    val dataNMT = loadDataNMT(batchSize, numSteps, 600)
    val dataset: ArrayDataset = dataNMT.first
    val srcVocab: Vocab = dataNMT.second.first
    val tgtVocab: Vocab = dataNMT.second.second
    println("Target: ${manager.create(floatArrayOf(tgtVocab.getIdx("<bos>").toFloat())).expandDims(0)}")

    encoder = Seq2SeqEncoder(srcVocab.length(), embedSize, numHiddens, numLayers, dropout)
    val decoder1 = Seq2SeqDecoder(tgtVocab.length(), embedSize, numHiddens, numLayers, dropout)

    val net = EncoderDecoder(encoder, decoder1)
    trainSeq2Seq(net, dataset, lr, numEpochs, tgtVocab, device)

    val engs = arrayOf("go .", "i lost .", "he's calm .", "i'm home .")
    val fras = arrayOf("va !", "j'ai perdu .", "il est calme .", "je suis chez moi .")
    for (i in engs.indices) {
        val pair = predictSeq2SeqLocal(net, engs[i], srcVocab, tgtVocab, numSteps, false, device)
        val translation: String = pair.first
        println("%s => %s, bleu %.3f".format(engs[i], translation, bleu(translation, fras[i], 2)))
    }
}

/**
 * Predict a sequence using a sequence-to-sequence model.
 */
private fun predictSeq2SeqLocal(
    net: EncoderDecoder,
    srcSentence: String,
    srcVocab: Vocab,
    tgtVocab: Vocab,
    numSteps: Int,
    saveAttentionWeights: Boolean,
    device: Device,
): Pair<String, List<NDArray?>> {
    val manager = NDManager.newBaseManager(device)
    val srcTokens =
        srcVocab.getIdxs(srcSentence.lowercase(Locale.getDefault()).split(" ")) +
            listOf(srcVocab.getIdx("<eos>"))
    val encValidLen = manager.create(longArrayOf(srcTokens.size.toLong()))
    val truncateSrcTokens = NMT.truncatePad(srcTokens, numSteps, srcVocab.getIdx("<pad>"))
    // Add the batch axis
    val encX = manager.create(truncateSrcTokens.toIntArray()).expandDims(0)
    val encOutputs = net.encoder.forward(ParameterStore(manager, false), NDList(encX, encValidLen), false)
    var decState = net.decoder.initState(encOutputs)
    // Add the batch axis
    var decX = manager.create(floatArrayOf(tgtVocab.getIdx("<bos>").toFloat())).expandDims(0)
    val outputSeq: MutableList<Int> = mutableListOf()
    val attentionWeightSeq: MutableList<NDArray?> = mutableListOf()
    for (i in 0 until numSteps) {
        val output =
            net.decoder.forward(
                ParameterStore(manager, false),
                NDList(decX).addAll(decState),
                false,
            )
        val decoderOutput = output[0]
        decState = output.subNDList(1)
        // We use the token with the highest prediction likelihood as the input
        // of the decoder at the next time step
        decX = decoderOutput.argMax(2)
        val pred = decX.squeeze(0).getLong().toInt()
        // Save attention weights (to be covered later)
        if (saveAttentionWeights) {
            attentionWeightSeq.add(net.decoder.attentionWeights)
        }
        // Once the end-of-sequence token is predicted, the generation of the
        // output sequence is complete
        if (pred == tgtVocab.getIdx("<eos>")) {
            break
        }
        outputSeq.add(pred)
    }
    val outputString: String = tgtVocab.toTokens(outputSeq).joinToString(separator = " ")
    return Pair(outputString, attentionWeightSeq.toList())
}
