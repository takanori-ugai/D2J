package jp.live.ugai.d2j

import ai.djl.Model
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator

fun main() {
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "false")
    System.setProperty("org.slf4j.simpleLogger.showLogName", "true")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.ndarray.index", "ERROR")
    System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN")

    val manager = NDManager.newBaseManager()
    val T = 1000L // Generate a total of 1000 points

    val time = manager.arange(1f, (T + 1).toFloat())
    val x = time.mul(0.01).sin().add(
        manager.randomNormal(0f, 0.2f, Shape(T), DataType.FLOAT32)
    )
    val tau = 4L
    val features = manager.zeros(Shape(T - tau, tau))

    for (i in 0 until tau) {
        features[NDIndex(":, {}", i)] = x[NDIndex("{}:{}", i, T - tau + i)]
    }
    val labels: NDArray = x[NDIndex("$tau:")].reshape(Shape(-1, 1))

    val batchSize = 16
    val nTrain = 600
// Only the first `nTrain` examples are used for training
// Only the first `nTrain` examples are used for training
    val trainIter = ArrayDataset.Builder()
        .setData(features[NDIndex(":{}", nTrain)])
        .optLabels(labels[NDIndex(":{}", nTrain)])
        .setSampling(batchSize, true)
        .build()

    val net = getNet()
    val model = train(net, trainIter, batchSize, 5, 0.01f)

    val translator = NoopTranslator(null)
    val predictor = model.newPredictor(translator)

    val onestepPreds = predictor.predict(NDList(features))[0]
    println(onestepPreds.get(NDIndex(":10")))

    val multiStepPreds = manager.zeros(Shape(T))
    multiStepPreds[NDIndex(":{}", nTrain + tau)] = x[NDIndex(":{}", nTrain + tau)]
    for (i in nTrain + tau until T) {
        val tempX = multiStepPreds[NDIndex("{}:{}", i - tau, i)].reshape(Shape(1, -1))
        val prediction = (predictor.predict(NDList(tempX)) as NDList)[0]
        multiStepPreds[NDIndex(i)] = prediction
    }
}

// var trainer: Trainer? = null

fun train(net: SequentialBlock, dataset: ArrayDataset, batchSize: Int, numEpochs: Int, learningRate: Float): Model {
    // Square Loss
    val loss: Loss = Loss.l2Loss()
    val lrt: Tracker = Tracker.fixed(learningRate)
    val adam: Optimizer = Optimizer.adam().optLearningRateTracker(lrt).build()
    val config = DefaultTrainingConfig(loss)
        .optOptimizer(adam) // Optimizer (loss function)
        .optInitializer(XavierInitializer(), "")
        .addTrainingListeners(*TrainingListener.Defaults.logging()) // Logging
    val model: Model = Model.newInstance("sequence")
    model.block = net
    val trainer = model.newTrainer(config)
    for (epoch in 1..numEpochs) {
        // Iterate over dataset
        for (batch in trainer.iterateDataset(dataset)) {
            // Update loss and evaulator
            EasyTrain.trainBatch(trainer, batch)

            // Update parameters
            trainer.step()
            batch.close()
        }

        // reset training and validation evaluators at end of epoch
        trainer.notifyListeners { listener: TrainingListener ->
            listener.onEpoch(trainer)
        }
        println("Epoch %d".format(epoch))
        println("Loss %f".format(trainer.trainingResult.trainLoss))
    }
    return model
}

fun getNet(): SequentialBlock {
    val net = SequentialBlock()
    net.add(Linear.builder().setUnits(10).build())
    net.add(Activation::relu)
    net.add(Linear.builder().setUnits(1).build())
    return net
}

class Sequence
