package jp.live.ugai.d2j.util

import ai.djl.Model
import ai.djl.basicdataset.tabular.AirfoilRandomAccess
import ai.djl.engine.Engine
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.Trainer
import ai.djl.training.dataset.Batch
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import jp.live.ugai.d2j.util.Training.linreg
import jp.live.ugai.d2j.util.Training.squaredLoss
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.letsPlot

/**
 * Singleton for TrainingChapter11.
 */
object TrainingChapter11 {
    /**
     * Executes getDataCh11.
     */
    fun getDataCh11(
        batchSize: Int,
        n: Int,
    ): AirfoilRandomAccess {
        // Load data
        val airfoil =
            AirfoilRandomAccess
                .builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, true)
                .optNormalize(true)
                .optLimit(n.toLong())
                .build()
        // Prepare Data
        airfoil.prepare()
        return airfoil
    }

    /**
     * Executes evaluateLoss.
     */
    fun evaluateLoss(
        dataIterator: Iterable<Batch>,
        w: NDArray,
        b: NDArray,
    ): Float {
        val metric = Accumulator(2) // sumLoss, numExamples
        for (batch in dataIterator) {
            val X = batch.data.head()
            val y = batch.labels.head()
            val yHat = linreg(X, w, b)
            val lossSum = squaredLoss(yHat, y).sum().getFloat()
            metric.add(floatArrayOf(lossSum, y.size().toFloat()))
            batch.close()
        }
        return metric.get(0) / metric.get(1)
    }

    /**
     * Executes plotLossEpoch.
     */
    fun plotLossEpoch(
        loss: List<Number>,
        epoch: List<Number>,
    ): Plot {
        var data =
            mapOf(
                "epoch" to epoch,
                "loss" to loss,
            )
        var plot = letsPlot()
        plot +=
            geomLine(data = data) {
                x = "epoch"
                y = "loss"
            }
        return (plot + ggsize(500, 400))
    }

    /**
     * Executes trainCh11.
     */
    fun trainCh11(
        trainer: (NDList, NDList, MutableMap<String, Float>) -> Unit,
        states: NDList,
        hyperparams: MutableMap<String, Float>,
        dataset: AirfoilRandomAccess,
        featureDim: Int,
        numEpochs: Int,
    ): LossTime {
        val manager = NDManager.newBaseManager()
        val w = manager.randomNormal(0f, 0.01f, Shape(featureDim.toLong(), 1L), DataType.FLOAT32)
        val b = manager.zeros(Shape(1))
        w.setRequiresGradient(true)
        b.setRequiresGradient(true)
        val params = NDList(w, b)
        var n = 0
        val stopWatch = StopWatch()
        stopWatch.start()
        var lastLoss = -1.0f
        val loss = mutableListOf<Float>()
        val epoch = ArrayList<Double>()
        repeat(numEpochs) {
            for (batch in dataset.getData(manager)) {
                val len = dataset.size().toInt() / batch.size // number of batches
                val X = batch.data.head()
                val y = batch.labels.head()
                val gc = Engine.getInstance().newGradientCollector()
                val yHat = linreg(X, params[0], params[1])
                var l = squaredLoss(yHat, y).mean()
                gc.backward(l)
                gc.close()
                trainer(params, states, hyperparams)
                n += X.shape[0].toInt()
                if (n % 200 == 0) {
                    stopWatch.stop()
                    lastLoss = evaluateLoss(dataset.getData(manager), params[0], params[1])
                    loss.add(lastLoss)
                    val lastEpoch = 1.0 * n / X.shape[0] / len
                    epoch.add(lastEpoch)
                    stopWatch.start()
                }
                batch.close()
            }
        }
//        plotLossEpoch(loss, epoch)
        System.out.printf("loss: %.3f, %.3f sec/epoch\n", lastLoss, stopWatch.avg())
        return LossTime(epoch, loss, stopWatch.cumsum())
    }

    /**
     * Executes trainConciseCh11.
     */
    fun trainConciseCh11(
        sgd: Optimizer?,
        dataset: AirfoilRandomAccess,
        numEpochs: Int,
    ): LossTime {
        // Initialization
        val manager = NDManager.newBaseManager()
        val net = SequentialBlock()
        val linear = Linear.builder().setUnits(1).build()
        net.add(linear)
        net.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
        val model = Model.newInstance("concise implementation")
        model.block = net
        val loss: Loss = Loss.l2Loss()
        val config =
            DefaultTrainingConfig(loss)
                .optOptimizer(sgd)
                .optDevices(manager.engine.getDevices(1)) // single GPU
                .addEvaluator(Accuracy()) // Model Accuracy
                .also { cfg ->
                    TrainingListener.Defaults.logging().forEach { cfg.addTrainingListeners(it) }
                } // Logging
        val trainer: Trainer = model.newTrainer(config)
        var n = 0
        val stopWatch = StopWatch()
        stopWatch.start()
        trainer.initialize(Shape(10, 5))
        val metrics = Metrics()
        trainer.metrics = metrics
        var lastLoss = -1f
        val lossArray = mutableListOf<Double>()
        val epochArray = mutableListOf<Double>()
        repeat(numEpochs) {
            for (batch in trainer.iterateDataset(dataset)) {
                val len = dataset.size().toInt() / batch.size // number of batches
                val X = batch.data.head()
                EasyTrain.trainBatch(trainer, batch)
                trainer.step()
                n += X.shape[0].toInt()
                if (n % 200 == 0) {
                    stopWatch.stop()
                    lastLoss =
                        evaluateLoss(
                            dataset.getData(manager),
                            linear.parameters[0]
                                .value
                                .array
                                .reshape(Shape(dataset.columnNames.size.toLong(), 1L)),
                            linear.parameters[1].value.array,
                        )
                    lossArray.add(lastLoss.toDouble())
                    val lastEpoch = 1.0 * n / X.shape[0] / len
                    epochArray.add(lastEpoch)
                    stopWatch.start()
                }
                batch.close()
            }
        }
//        plotLossEpoch(lossArray, epochArray)
        System.out.printf("loss: %.3f, %.3f sec/epoch\n", lastLoss, stopWatch.avg())
        return LossTime(epochArray, lossArray, stopWatch.cumsum())
    } // End Ch11 Optimization
}

/**
 * Represents LossTime.
 * @property epoch The epoch.
 * @property loss The loss.
 * @property time The time.
 */
class LossTime(
    /**
     * The epoch.
     */
    val epoch: List<Number>,
    /**
     * The loss.
     */
    val loss: List<Number>,
    /**
     * The time.
     */
    val time: List<Double>,
)
