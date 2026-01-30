package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.Dropout
import ai.djl.nn.norm.LayerNorm
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.ParameterStore
import ai.djl.training.Trainer
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList
import jp.live.ugai.d2j.attention.MultiHeadAttention

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

    val manager = NDManager.newBaseManager()
    val ps = ParameterStore(manager, false)
    val imgSize = 96
    val patchSize = 16
    val numHiddens = 512
    val batchSize = 4
    val patchEmb = PatchEmbedding(imgSize, patchSize, numHiddens)
    val inputBatch = manager.randomNormal(Shape(batchSize.toLong(), 3, imgSize.toLong(), imgSize.toLong()))
    patchEmb.initialize(manager, DataType.FLOAT32, inputBatch.shape)
    println(patchEmb.forward(ps, NDList(inputBatch), false))
//        (batch_size, (img_size//patch_size)**2, num_hiddens))

    val sampleInput = manager.ones(Shape(2, 100, 24))
    val encoderBlk = ViTBlock(24, 24, 48, 8, 0.5f)
    encoderBlk.initialize(manager, DataType.FLOAT32, sampleInput.shape, Shape(2))
    println(encoderBlk.forward(ps, NDList(sampleInput, null), false))
    println("Shapes : ${encoderBlk.getOutputShapes(arrayOf(sampleInput.shape)).toList()}")

    val imgSize0 = 28
    val patchSize0 = 16
    val numHiddens0 = 512
    val mlpNumHiddens0 = 2048
    val numHeads0 = 8
    val numBlks0 = 2
    val embDropout0 = 0.1f
    val blkDropout0 = 0.1f
    val batchSize0 = 256
    val lr = 0.001f
    val trainImages = manager.ones(Shape(batchSize0.toLong(), 1, imgSize0.toLong(), imgSize0.toLong()))
    val encoder =
        ViT(
            imgSize0,
            patchSize0,
            numHiddens0,
            mlpNumHiddens0,
            numHeads0,
            numBlks0,
            embDropout0,
            blkDropout0,
        )
//    encoder.initialize(manager, DataType.FLOAT32, X0.shape)

    val randomShuffle = true

// Get Training and Validation Datasets

// Get Training and Validation Datasets
    val trainingSet =
        FashionMnist
            .builder()
            .optUsage(Dataset.Usage.TRAIN)
            .setSampling(batchSize0, randomShuffle)
            .optLimit(Long.MAX_VALUE)
            .build()

    val validationSet =
        FashionMnist
            .builder()
            .optUsage(Dataset.Usage.TEST)
            .setSampling(batchSize0, false)
            .optLimit(Long.MAX_VALUE)
            .build()
    trainingSet.prepare()
    validationSet.prepare()
    val model: Model = Model.newInstance("softmax-regression")
    model.setBlock(encoder)
    val loss: Loss = Loss.softmaxCrossEntropyLoss()
    val lrt: Tracker = Tracker.fixed(lr)
    val adam: Optimizer = Optimizer.adam().optLearningRateTracker(lrt).build()
    val config: DefaultTrainingConfig =
        DefaultTrainingConfig(loss)
            .optOptimizer(adam) // Optimizer (loss function)
            .optInitializer(XavierInitializer(), "")
            .addEvaluator(Accuracy()) // Model Accuracy
            .also { cfg ->
                TrainingListener.Defaults.logging().forEach { cfg.addTrainingListeners(it) }
            }

    val trainer: Trainer = model.newTrainer(config)
    trainer.initialize(trainImages.shape)
    trainer.metrics = Metrics()
    EasyTrain.fit(trainer, 5, trainingSet, validationSet)

    val batch = validationSet.getData(manager).iterator().next()
    val batchImages = batch.getData().head()
    val yHat: IntArray =
        encoder
            .forward(ps, NDList(batchImages), false)
            .head()
            .argMax(1)
            .toType(DataType.INT32, false)
            .toIntArray()
    println(yHat.toList().subList(0, 20))
    println(
        batch
            .getLabels()
            .head()
            .toType(DataType.INT32, false)
            .toIntArray()
            .toList()
            .subList(0, 20),
    )
}

/**
 * Represents PatchEmbedding.
 * @property patchSize The patchSize.
 * @property numHiddens The numHiddens.
 */
class PatchEmbedding(
    imgSize: Int = 96,
    /**
     * The patchSize.
     */
    val patchSize: Int = 16,
    /**
     * The numHiddens.
     */
    val numHiddens: Int = 512,
) : AbstractBlock() {
    /**
     * The numPatches.
     */
    val numPatches = (imgSize / patchSize) * (imgSize / patchSize)

    /**
     * The conv.
     */
    val conv =
        Conv2d
            .builder()
            .setKernelShape(Shape(patchSize.toLong(), patchSize.toLong()))
            .optStride(Shape(patchSize.toLong(), patchSize.toLong()))
            .setFilters(numHiddens)
            .build()

    init {
        require(imgSize % patchSize == 0) {
            "imgSize ($imgSize) must be divisible by patchSize ($patchSize)"
        }
        addChildBlock("conv", conv)
    }

    /**
     * Executes forwardInternal.
     */
    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        // Output shape: (batch size, no. of patches, no. of channels)
        val f = conv.forward(parameterStore, inputs, training, params).head()
        return NDList(f.reshape(Shape(f.shape[0], f.shape[1], -1)).transpose(0, 2, 1))
    }

    /**
     * Executes initializeChildBlocks.
     */
    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        require(inputShapes.size == 1) {
            "PatchEmbedding expects a single input shape, got ${inputShapes.size}."
        }
        conv.initialize(manager, dataType, inputShapes[0])
    }

    /**
     * Executes getOutputShapes.
     */
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> =
        arrayOf(Shape(inputShapes[0][0], numPatches.toLong(), numHiddens.toLong()))
}

/**
 * Represents ViTBlock.
 * @property normShape The normShape.
 */
class ViTBlock(
    numHiddens: Int,
    /**
     * The normShape.
     */
    val normShape: Int,
    mlpNumHiddens: Int,
    numHeads: Int,
    dropout: Float,
    useBias: Boolean = false,
) : AbstractBlock() {
    val ln1 = LayerNorm.builder().build()
    val attention = MultiHeadAttention(numHiddens, numHeads, dropout, useBias)
    val ln2 = LayerNorm.builder().build()
    val mlp = ViTMLP(mlpNumHiddens, numHiddens, dropout)

    init {
        addChildBlock("ln1", ln1)
        addChildBlock("attention", attention)
        addChildBlock("ln2", ln2)
        addChildBlock("mlp", mlp)
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        require(inputs.isNotEmpty()) { "ViTBlock requires at least one input." }
        val input = inputs[0]
        val norm1 = ln1.forward(parameterStore, NDList(input), training, params).head()
        val att = attention.forward(parameterStore, NDList(norm1, norm1, norm1), training, params).head()
        val residual = input.add(att)
        val norm2 = ln2.forward(parameterStore, NDList(residual), training, params).head()
        val mlpOut = mlp.forward(parameterStore, NDList(norm2), training, params).head()
        return NDList(residual.add(mlpOut))
    }

    override fun initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        vararg inputShapes: Shape,
    ) {
        ln1.initialize(manager, dataType, inputShapes[0])
        attention.initialize(manager, dataType, inputShapes[0], inputShapes[0], inputShapes[0])
        ln2.initialize(manager, dataType, inputShapes[0])
        mlp.initialize(manager, dataType, inputShapes[0])
    }

    // We won't implement this since we won't be using it but it's required as part of an AbstractBlock
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> = inputShapes
}

/**
 * Represents ViTMLP.
 */
class ViTMLP(
    mlpNumHiddens: Int,
    mlpNumOutputs: Int,
    dropout: Float = 0.5f,
) : SequentialBlock() {
    init {
        add(Linear.builder().setUnits(mlpNumHiddens.toLong()).build())
        add(Activation::relu)
        add(Dropout.builder().optRate(dropout).build())
        add(Linear.builder().setUnits(mlpNumOutputs.toLong()).build())
        add(Dropout.builder().optRate(dropout).build())
    }
}

/**
 * Placeholder for a Vision Transformer example container.
 */
internal class VisionTransformer
