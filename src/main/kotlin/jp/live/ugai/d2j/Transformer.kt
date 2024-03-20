package jp.live.ugai.d2j

import ai.djl.Device
import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.modality.nlp.Vocabulary
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.BatchNorm
import ai.djl.nn.norm.Dropout
import ai.djl.nn.norm.LayerNorm
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import jp.live.ugai.d2j.attention.AttentionDecoder
import jp.live.ugai.d2j.attention.MultiHeadAttention
import jp.live.ugai.d2j.lstm.Encoder
import jp.live.ugai.d2j.lstm.EncoderDecoder
import jp.live.ugai.d2j.timemachine.Vocab
import jp.live.ugai.d2j.util.Accumulator
import jp.live.ugai.d2j.util.NMT
import jp.live.ugai.d2j.util.StopWatch
import jp.live.ugai.d2j.util.TrainingChapter9
import java.util.Locale

fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    val manager = NDManager.newBaseManager()
    val ps = ParameterStore(manager, false)

    fun positionWiseFFN(ffn_num_hiddens: Long, ffn_num_outputs: Long): AbstractBlock {
        val net = SequentialBlock()
        net.add(Linear.builder().setUnits(ffn_num_hiddens).build())
        net.add(Activation::relu)
        net.add(Linear.builder().setUnits(ffn_num_outputs).build())
        return net
    }
//    net.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
    val ffn0 = positionWiseFFN(4, 8)
    ffn0.initialize(manager, DataType.FLOAT32, Shape(2, 3, 4))
    println(ffn0.forward(ps, NDList(manager.ones(Shape(2, 3, 4))), false)[0][0])

    val ln0 = LayerNorm.builder().build()
    ln0.initialize(manager, DataType.FLOAT32, Shape(2, 2))
    val bn0 = BatchNorm.builder().build()
    bn0.initialize(manager, DataType.FLOAT32, Shape(2, 2))
    val X0 = manager.create(floatArrayOf(1f, 2f, 2f, 3f)).reshape(Shape(2, 2))
    print("LayerNorm: ")
    println(ln0.forward(ps, NDList(X0), false)[0])
    print("BatchNorm: ")
    println(bn0.forward(ps, NDList(X0), false)[0])

    class AddNorm(rate: Float) : AbstractBlock() {
        val dropout = Dropout.builder().optRate(rate).build()
        val ln = LayerNorm.builder().build()

        init {
            addChildBlock("dropout", dropout)
            addChildBlock("layerNorm", ln)
        }

        override fun forwardInternal(
            ps: ParameterStore,
            inputs: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            val x = inputs[0]
            val y = inputs[1]
            val dropoutResult = dropout.forward(ps, NDList(y), training, params).singletonOrThrow()
            val result = ln.forward(ps, NDList(dropoutResult.add(x)), training, params)
            return result
        }

        override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
            return inputShapes
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
            dropout.initialize(manager, dataType, *inputShapes)
            ln.initialize(manager, dataType, *inputShapes)
        }
    }

    val addNorm = AddNorm(0.5f)
    addNorm.initialize(manager, DataType.FLOAT32, Shape(2, 3, 4))
    println(addNorm.forward(ps, NDList(manager.ones(Shape(2, 3, 4)), manager.ones(Shape(2, 3, 4))), false)[0].shapeEquals(manager.ones(Shape(2, 3, 4))))

    class TransformerEncoderBlock(
        numHiddens: Int,
        ffnNumHiddens: Long,
        numHeads: Int,
        dropout: Float,
        useBias: Boolean = false
    ) : AbstractBlock() {
        val attention = MultiHeadAttention(numHiddens, numHeads, dropout, useBias)
        val addnorm1 = AddNorm(dropout)
        val ffn = positionWiseFFN(ffnNumHiddens, numHiddens.toLong())
        val addnorm2 = AddNorm(dropout)
        init {
            addChildBlock("attention", attention)
            addChildBlock("addnorm1", addnorm1)
            addChildBlock("ffn", ffn)
            addChildBlock("addnorm2", addnorm2)
        }

        override fun forwardInternal(
            ps: ParameterStore,
            inputs: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            val x = inputs[0]
            val validLens = inputs[1]
            val y = addnorm1.forward(
                ps,
                NDList(x, attention.forward(ps, NDList(x, x, x, validLens), training, params).singletonOrThrow()),
                training,
                params
            )
            val ret = addnorm2.forward(
                ps,
                NDList(y.singletonOrThrow(), ffn.forward(ps, y, training, params).singletonOrThrow()),
                training,
                params
            )
            return ret
        }

        override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
            return arrayOf<Shape>()
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
            val shapes = arrayOf(inputShapes[0], inputShapes[0], inputShapes[0], inputShapes[1])
            attention.initialize(manager, dataType, *shapes)
            addnorm1.initialize(manager, dataType, inputShapes[0])
            ffn.initialize(manager, dataType, inputShapes[0])
            addnorm2.initialize(manager, dataType, inputShapes[0])
        }
    }

    val X1 = manager.ones(Shape(2, 100, 24))
    val validLens = manager.create(floatArrayOf(3f, 2f))
    val encoderBlock = TransformerEncoderBlock(24, 48, 8, 0.5f)
    encoderBlock.initialize(manager, DataType.FLOAT32, X1.shape, validLens.shape)
    println(encoderBlock.forward(ps, NDList(X1, validLens), false))

    class TransformerEncoder(
        vocabSize: Int,
        val numHiddens: Int,
        ffnNumHiddens: Long,
        numHeads: Long,
        numBlks: Int,
        dropout: Float,
        useBias: Boolean = false
    ) : Encoder() {

        private val embedding: TrainableWordEmbedding
        val posEncoding = PositionalEncoding(numHiddens, dropout, 1000, manager)
        val blks = mutableListOf<TransformerEncoderBlock>()
        val attentionWeights = Array<NDArray?>(numBlks) { null }

        /* The RNN encoder for sequence to sequence learning. */
        init {
            val list: List<String> = (0 until vocabSize).map { it.toString() }
            val vocab: Vocabulary = DefaultVocabulary(list)
            // Embedding layer
            embedding = TrainableWordEmbedding.builder()
                .optNumEmbeddings(vocabSize)
                .setEmbeddingSize(numHiddens)
                .setVocabulary(vocab)
                .build()
            addChildBlock("embedding", embedding)
            repeat(numBlks) {
                blks.add(TransformerEncoderBlock(numHiddens, ffnNumHiddens, numHeads.toInt(), dropout, useBias))
            }
        }

        override fun forwardInternal(
            ps: ParameterStore,
            inputs: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            var X = inputs[0]
            val validLens = inputs[1]
            val emb = embedding
                .forward(ps, NDList(X), training, params)
                .singletonOrThrow()
                .mul(Math.sqrt(numHiddens.toDouble()))
            X = posEncoding.forward(ps, NDList(emb), training, params).singletonOrThrow()
            for (i in 0 until blks.size) {
                X = blks[i].forward(ps, NDList(X, validLens), training, params).singletonOrThrow()
                attentionWeights[i] = blks[i].attention.attention.attentionWeights
            }
            return NDList(X, validLens)
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
            embedding.initialize(manager, dataType, *inputShapes)
            for (blk in blks) {
                blk.initialize(manager, dataType, inputShapes[0].add(numHiddens.toLong()), inputShapes[1])
            }
        }
    }

    val encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5f)
    encoder.initialize(manager, DataType.FLOAT32, Shape(2, 100), validLens.shape)
    println(encoder.forward(ps, NDList(manager.ones(Shape(2, 100)), validLens), false))

    class TransformerDecoderBlock(
        val numHiddens: Int,
        ffnNumHiddens: Long,
        numHeads: Long,
        dropout: Float,
        _i: Int
    ) : AbstractBlock() {
        val i = _i
        val attention1 = MultiHeadAttention(numHiddens, numHeads.toInt(), dropout, false)
        val addnorm1 = AddNorm(dropout)
        val attention2 = MultiHeadAttention(numHiddens, numHeads.toInt(), dropout, false)
        val addnorm2 = AddNorm(dropout)
        val ffn = positionWiseFFN(ffnNumHiddens, numHiddens.toLong())
        val addnorm3 = AddNorm(dropout)

        init {
            addChildBlock("attention1", attention1)
            addChildBlock("addnorm1", addnorm1)
            addChildBlock("attention2", attention2)
            addChildBlock("addnorm2", addnorm2)
            addChildBlock("ffn", ffn)
            addChildBlock("addnorm3", addnorm3)
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
            val shapes1 = arrayOf(inputShapes[0], inputShapes[0], inputShapes[0], inputShapes[1])
            attention1.initialize(manager, dataType, *shapes1)
            addnorm1.initialize(manager, dataType, inputShapes[0])
            attention2.initialize(manager, dataType, *shapes1)
            addnorm2.initialize(manager, dataType, inputShapes[0])
            ffn.initialize(manager, dataType, inputShapes[0])
            addnorm3.initialize(manager, dataType, inputShapes[0])
        }
        override fun forwardInternal(
            ps: ParameterStore,
            inputs: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            val input0 = inputs[0]
            val encOutputs = inputs[1]
            val envValidLens = inputs[2]
//        # During training, all the tokens of any output sequence are processed
//        # at the same time, so state[2][self.i] is None as initialized. When
//        # decoding any output sequence token by token during prediction,
//        # state[2][self.i] contains representations of the decoded output at
//        # the i-th block up to the current time step

            // TODO FIX IT
//            if state[2][self.i] is None:
//            key_values = X
//            else:
//            key_values = torch.cat((state[2][self.i], X), dim=1)
//            state[2][self.i] = key_values

            var keyValues: NDArray?
            if (inputs.size < 4 || inputs[3] == null) {
                keyValues = inputs[0]
            } else {
                keyValues = inputs[3]
            }
            /*
            } else if (inputs[3]!!.size(0) < i.toLong()) {
                keyValues = inputs[3].concat(inputs[0])
//                keyValues = inputs[3]
            } else {
//                println(inputs[3].get(i.toLong()).concat(input0))
//                val keyValue = inputs[3].get(i.toLong()).concat(input0, 1)
                keyValues = inputs[3].get(i.toLong()).concat(input0)
//                keyValues!!.set(NDIndex(i.toLong()), keyValue)
//                if (training) {
//                    keyValues = keyValue.expandDims(0)
//                } else {
//                    keyValues = inputs[3].concat(input0.expandDims(0))
//                }
            }
            println("KEYVALUES:: $keyValues")

             */

            var decValidLens: NDArray?
            if (training) {
                val batchSize = input0.shape[0]
                val numSteps = input0.shape[1]
                //  Shape of dec_valid_lens: (batch_size, num_steps), where every
                //  row is [1, 2, ..., num_steps]
                decValidLens = manager.arange(1f, (numSteps + 1).toFloat()).reshape(1, numSteps).repeat(0, batchSize)
            } else {
                decValidLens = null
            }
//        # Self-attention
            val X2 = attention1.forward(ps, NDList(input0, keyValues, keyValues, decValidLens), training)
            val Y = addnorm1.forward(ps, NDList(input0, X2.head()), training)
//        # Encoder-decoder attention. Shape of enc_outputs:
//        # (batch_size, num_steps, num_hiddens)
            val Y2 = attention2.forward(ps, NDList(Y.head(), encOutputs, encOutputs, envValidLens), training)
            val Z = addnorm2.forward(ps, NDList(Y.head(), Y2.head()), training)
            return NDList(
                addnorm3.forward(ps, NDList(Z.head(), ffn.forward(ps, NDList(Z), training).head()), training).head(),
                encOutputs,
                envValidLens,
                keyValues
            )
        }

        override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
            return arrayOf<Shape>()
        }
    }

    val decoderBlk = TransformerDecoderBlock(24, 48, 8, 0.5f, 0)
    val X = manager.ones(Shape(2, 100, 24))
    val input = NDList(X, validLens)

    decoderBlk.initialize(manager, DataType.FLOAT32, *input.shapes)
    val state = encoderBlock.forward(ps, NDList(X, validLens, validLens), false)
    println(decoderBlk.forward(ps, NDList(X, state.head(), validLens, null), false))

    class TransformerDecoder(
        vocabSize: Int,
        val numHiddens: Int,
        ffnNumHiddens: Int,
        numHeads: Int,
        val numBlks: Int,
        dropout: Float
    ) : AttentionDecoder() {
        val list: List<String> = (0 until vocabSize).map { it.toString() }
        val vocab: Vocabulary = DefaultVocabulary(list)
        val embedding = TrainableWordEmbedding.builder()
            .optNumEmbeddings(vocabSize)
            .setEmbeddingSize(numHiddens)
            .setVocabulary(vocab)
            .build()
        val posEncoding = PositionalEncoding(numHiddens, dropout, 1000, manager)
        val blks = mutableListOf<TransformerDecoderBlock>()

        //            val attentionWeights = Array<NDArray?>(numBlks) { null }
        val linear = Linear.builder().setUnits(vocabSize.toLong()).build()
        var attentionWeightsArr2: MutableList<NDArray?>? = null
        var attentionWeightsArr1: MutableList<NDArray?>? = null

        init {
            addChildBlock("embedding", embedding)
            repeat(numBlks) {
                blks.add(
                    TransformerDecoderBlock(
                        numHiddens,
                        ffnNumHiddens.toLong(),
                        numHeads.toLong(),
                        dropout,
                        it
                    )
                )
            }
        }

        override fun initState(input: NDList): NDList {
            val encOutputs = input[0]
            val encValidLens = input[1]
            return NDList(encOutputs, encValidLens, null)
        }

        override fun forwardInternal(
            ps: ParameterStore,
            inputs: NDList,
            training: Boolean,
            params: PairList<String, Any>?
        ): NDList {
            var X = inputs[0]
            val state = inputs.subNDList(1)
            val pos = posEncoding.forward(
                ps,
                NDList(embedding.forward(ps, NDList(X), training, params).head().mul(Math.sqrt(numHiddens.toDouble()))),
                training,
                params
            )
            attentionWeightsArr1 = mutableListOf()
            attentionWeightsArr2 = mutableListOf()
            for (i in 0 until blks.size) {
                val blk = blks[i].forward(ps, NDList(pos.head()).addAll(state), training, params)
                attentionWeightsArr1!!.add(blks[i].attention1.attention.attentionWeights)
                attentionWeightsArr2!!.add(blks[i].attention2.attention.attentionWeights)
            }
            var ret = linear.forward(ps, NDList(pos.head()), training, params)
            return NDList(ret.head()).addAll(state)
        }

        override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
            embedding.initialize(manager, dataType, *inputShapes)
            posEncoding.initialize(manager, dataType, *inputShapes)
            for (blk in blks) {
                blk.initialize(manager, dataType, inputShapes[0], inputShapes[1])
            }
        }
    }

    fun train() {
//        num_hiddens, num_blks, dropout = 256, 2, 0.2
//        ffn_num_hiddens, num_heads = 64, 4

        val numHiddens = 256
        val numBlks = 2
        val ffnNumHiddens = 64
        val numHeads = 4
        val batchSize = 2
        val numEpochs = 30
        val numSteps = 35
        println(numSteps)

        val dropout = 0.2f
        val lr = 0.001f
        val device = manager.device

        val dataNMT = NMT.loadDataNMT(batchSize, numSteps, 600)
        val dataset: ArrayDataset = dataNMT.first
        val srcVocab: Vocab = dataNMT.second.first
        val tgtVocab: Vocab = dataNMT.second.second

        val encoder = TransformerEncoder(
            srcVocab.length(),
            numHiddens,
            ffnNumHiddens.toLong(),
            numHeads.toLong(),
            numBlks,
            dropout
        )
        encoder.initialize(manager, DataType.FLOAT32, Shape(2, 35), Shape(2))

        val decoder = TransformerDecoder(tgtVocab.length(), numHiddens, ffnNumHiddens, numHeads, numBlks, dropout)
        decoder.initialize(manager, DataType.FLOAT32, Shape(2, 35, 256), Shape(2))

        val net = EncoderDecoder(encoder, decoder)
        fun trainSeq2Seq(
            net: EncoderDecoder,
            dataset: ArrayDataset,
            lr: Float,
            numEpochs: Int,
            tgtVocab: Vocab,
            device: Device
        ) {
            val loss: Loss = MaskedSoftmaxCELoss()
            val lrt: Tracker = Tracker.fixed(lr)
            val adam: Optimizer = Optimizer.adam().optLearningRateTracker(lrt).build()
            val config: DefaultTrainingConfig = DefaultTrainingConfig(loss)
                .optOptimizer(adam) // Optimizer (loss function)
                .optInitializer(XavierInitializer(), "")
            val model: Model = Model.newInstance("")
            model.block = net
            val trainer: Trainer = model.newTrainer(config)
//    val animator = Animator()
            var watch: StopWatch
            var metric: Accumulator
            var lossValue = 0.0
            var speed = 0.0
            for (epoch in 1..numEpochs) {
                watch = StopWatch()
                metric = Accumulator(2) // Sum of training loss, no. of tokens
                // Iterate over dataset
                for (batch in dataset.getData(manager)) {
                    val X: NDArray = batch.data.get(0)
                    val lenX: NDArray = batch.data.get(1)
                    val Y: NDArray = batch.labels.get(0)
                    val lenY: NDArray = batch.labels.get(1)
                    val bos: NDArray = manager
                        .full(Shape(Y.shape[0]), tgtVocab.getIdx("<bos>"))
                        .reshape(-1, 1)
                    val decInput: NDArray = NDArrays.concat(
                        NDList(bos, Y.get(NDIndex(":, :-1"))),
                        1
                    ) // Teacher forcing
                    Engine.getInstance().newGradientCollector().use { gc ->
                        val yHat: NDArray = net.forward(
                            ParameterStore(manager, false),
                            NDList(X, decInput, lenX),
                            true
                        )
                            .get(0)
                        val l = loss.evaluate(NDList(Y, lenY), NDList(yHat))
                        gc.backward(l)
                        metric.add(floatArrayOf(l.sum().getFloat(), lenY.sum().getLong().toFloat()))
                    }
                    TrainingChapter9.gradClipping(net, 1, manager)
                    // Update parameters
                    trainer.step()
                }
                lossValue = metric.get(0).toDouble() / metric.get(1)
                speed = metric.get(1) / watch.stop()
//                if ((epoch + 1) % 10 == 0) {
//            animator.add(epoch + 1, lossValue.toFloat(), "loss")
//            animator.show()
                println("${epoch + 1} : $lossValue")
//                }
            }
            println("loss: %.3f, %.1f tokens/sec on %s%n".format(lossValue, speed, device.toString()))
        }
        trainSeq2Seq(net, dataset, lr, numEpochs, tgtVocab, device)

        fun predictSeq2Seq(
            net: EncoderDecoder,
            srcSentence: String,
            srcVocab: Vocab,
            tgtVocab: Vocab,
            numSteps: Int,
            device: Device,
            saveAttentionWeights: Boolean
        ): Pair<String, List<NDArray?>> {
            val srcTokens = srcVocab.getIdxs(srcSentence.lowercase(Locale.getDefault()).split(" ")) +
                listOf(srcVocab.getIdx("<eos>"))
            val encValidLen = manager.create(srcTokens.size).reshape(1)
            val truncateSrcTokens = NMT.truncatePad(srcTokens, numSteps, srcVocab.getIdx("<pad>"))
            // Add the batch axis
            val encX = manager.create(truncateSrcTokens.toIntArray()).expandDims(0)
            val encOutputs = net.encoder.forward(ParameterStore(manager, false), NDList(encX, encValidLen), false)
            var decState = net.decoder.initState(encOutputs)
            // Add the batch axis
            var decX = manager.create(floatArrayOf(tgtVocab.getIdx("<bos>").toFloat())).repeat(35).expandDims(0)
            val outputSeq: MutableList<Int> = mutableListOf()
            val attentionWeightSeq: MutableList<NDArray?> = mutableListOf()
            for (i in 0 until numSteps) {
//                println(i)
                val output = net.decoder.forward(
                    ParameterStore(manager, false),
                    NDList(decX).addAll(decState),
                    false
                )

//                val encOutputs = encoder.forward(parameterStore, encX, training, params)
//                val decState = decoder.initState(encOutputs)
//                val inp = NDList(decX).addAll(decState)
//                return decoder.forward(parameterStore, inp, training, params)

                val Y = output[0]
                decState = output.subNDList(1)
                // We use the token with the highest prediction likelihood as the input
                // of the decoder at the next time step
//                println("Y:::$Y")
//                println("Y(1)::: ${Y.get(NDIndex("0,2")).argMax(0).getLong().toInt()}")
//                decX = Y.argMax(2)
//                println("DECX: ${decX.squeeze(0)}")
//                val pred = decX.squeeze(0).getLong().toInt()
                val pred = Y.get(NDIndex("0,2")).argMax(0).getLong().toInt()
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

        /* Compute the BLEU. */
        fun bleu(predSeq: String, labelSeq: String, k: Int): Double {
            val predTokens = predSeq.split(" ")
            val labelTokens = labelSeq.split(" ")
            val lenPred = predTokens.size
            val lenLabel = labelTokens.size
            var score = Math.exp(Math.min(0.toDouble(), 1.0 - lenLabel / lenPred))
            for (n in 1 until k + 1) {
                var numMatches = 0
                val labelSubs = mutableMapOf<String, Int>()
                for (i in 0 until lenLabel - n + 1) {
                    val key = labelTokens.subList(i, i + n).joinToString(separator = " ")
                    labelSubs.put(key, labelSubs.getOrDefault(key, 0) + 1)
                }
                for (i in 0 until lenPred - n + 1) {
                    // val key =predTokens.subList(i, i + n).joinToString(" ")
                    val key = predTokens.subList(i, i + n).joinToString(separator = " ")
                    if (labelSubs.getOrDefault(key, 0) > 0) {
                        numMatches += 1
                        labelSubs.put(key, labelSubs.getOrDefault(key, 0) - 1)
                    }
                }
                score *= Math.pow(numMatches.toDouble() / (lenPred - n + 1).toDouble(), Math.pow(0.5, n.toDouble()))
            }
            return score
        }

        val engs = arrayOf("go .", "i lost .", "he's calm .", "i'm home .")
        val fras = arrayOf("va !", "j'ai perdu .", "il est calme .", "je suis chez moi .")
        for (i in engs.indices) {
            val pair = predictSeq2Seq(net, engs[i], srcVocab, tgtVocab, numSteps, device, false)
            val translation: String = pair.first
            val attentionWeightSeq = pair.second
            println("%s => %s, bleu %.3f".format(engs[i], translation, bleu(translation, fras[i], 2)))
        }
    }

    train()
}

class Transformer
