package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.modality.cv.transform.Resize
import ai.djl.modality.cv.transform.ToTensor
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.Blocks
import ai.djl.nn.SequentialBlock
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.Dropout
import ai.djl.nn.pooling.Pool
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import jp.live.ugai.d2j.util.Training

fun getLong(
    nm: String,
    n: Long,
): Long {
    val name = System.getProperty(nm)
    return if (null == name) n.toLong() else name.toLong()
}

fun main() {
    val manager = NDManager.newBaseManager()

    val block = SequentialBlock()
// Here, we use a larger 11 x 11 window to capture objects. At the same time,
// we use a stride of 4 to greatly reduce the height and width of the output.
// Here, the number of output channels is much larger than that in LeNet
    block
        .add(
            Conv2d
                .builder()
                .setKernelShape(Shape(11, 11))
                .optStride(Shape(4, 4))
                .setFilters(96)
                .build(),
        ).add(Activation::relu)
        .add(Pool.maxPool2dBlock(Shape(3, 3), Shape(2, 2)))
        // Make the convolution window smaller, set padding to 2 for consistent
        // height and width across the input and output, and increase the
        // number of output channels
        .add(
            Conv2d
                .builder()
                .setKernelShape(Shape(5, 5))
                .optPadding(Shape(2, 2))
                .setFilters(256)
                .build(),
        ).add(Activation::relu)
        .add(Pool.maxPool2dBlock(Shape(3, 3), Shape(2, 2)))
        // Use three successive convolutional layers and a smaller convolution
        // window. Except for the final convolutional layer, the number of
        // output channels is further increased. Pooling layers are not used to
        // reduce the height and width of input after the first two
        // convolutional layers
        .add(
            Conv2d
                .builder()
                .setKernelShape(Shape(3, 3))
                .optPadding(Shape(1, 1))
                .setFilters(384)
                .build(),
        ).add(Activation::relu)
        .add(
            Conv2d
                .builder()
                .setKernelShape(Shape(3, 3))
                .optPadding(Shape(1, 1))
                .setFilters(384)
                .build(),
        ).add(Activation::relu)
        .add(
            Conv2d
                .builder()
                .setKernelShape(Shape(3, 3))
                .optPadding(Shape(1, 1))
                .setFilters(256)
                .build(),
        ).add(Activation::relu)
        .add(Pool.maxPool2dBlock(Shape(3, 3), Shape(2, 2)))
        // Here, the number of outputs of the fully connected layer is several
        // times larger than that in LeNet. Use the dropout layer to mitigate
        // overfitting
        .add(Blocks.batchFlattenBlock())
        .add(
            Linear
                .builder()
                .setUnits(4096)
                .build(),
        ).add(Activation::relu)
        .add(
            Dropout
                .builder()
                .optRate(0.5f)
                .build(),
        ).add(
            Linear
                .builder()
                .setUnits(4096)
                .build(),
        ).add(Activation::relu)
        .add(
            Dropout
                .builder()
                .optRate(0.5f)
                .build(),
        )
        // Output layer. Since we are using Fashion-MNIST, the number of
        // classes is 10, instead of 1000 as in the paper
        .add(Linear.builder().setUnits(10).build())

    val lr = 0.01f

    val model = Model.newInstance("cnn")
    model.block = block

    val loss = Loss.softmaxCrossEntropyLoss()

    val lrt = Tracker.fixed(lr)
    val sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config =
        DefaultTrainingConfig(loss)
            .optOptimizer(sgd) // Optimizer (loss function)
            .addEvaluator(Accuracy()) // Model Accuracy
            .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging

    val trainer = model.newTrainer(config)

    val X = manager.randomUniform(0f, 1.0f, Shape(1, 1, 224, 224))
    trainer.initialize(X.shape)

    var currentShape = X.shape

    for (i in 0 until block.children.size()) {
        val newShape =
            block.children
                .get(i)
                .value
                .getOutputShapes(arrayOf<Shape>(currentShape))
        currentShape = newShape[0]
        println(block.children.get(i).key + " layer output : " + currentShape)
    }

    val batchSize = 128
    val numEpochs = Integer.getInteger("MAX_EPOCH", 10)

// trainLoss;
// double[] testAccuracy;
// double[] epochCount;
// double[] trainAccuracy;

    val epochCount = DoubleArray(numEpochs) { it.toDouble() + 1 }

// for (int i = 0; i < epochCount.length; i++) {
//    epochCount[i] = (i + 1);//
// }

    val trainIter =
        FashionMnist
            .builder()
            .addTransform(Resize(224))
            .addTransform(ToTensor())
            .optUsage(Dataset.Usage.TRAIN)
            .setSampling(batchSize, true)
            .optLimit(getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build()

    val testIter =
        FashionMnist
            .builder()
            .addTransform(Resize(224))
            .addTransform(ToTensor())
            .optUsage(Dataset.Usage.TEST)
            .setSampling(batchSize, true)
            .optLimit(getLong("DATASET_LIMIT", Long.MAX_VALUE))
            .build()

    trainIter.prepare()
    testIter.prepare()

    val evaluatorMetrics = mutableMapOf<String, DoubleArray>()
    val avgTrainTimePerEpoch = Training.trainingChapter6(trainIter, testIter, numEpochs, trainer, evaluatorMetrics)
}

class AlexNet
