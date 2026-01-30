import ai.djl.Model
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Block
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Batch
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.translate.Batchifier

/**
 * Executes main.
 */
fun main() {
    val nIn = 1L
    val nOut = 2L
    val minibatch = 3

    // Create NDManager for memory management
    val manager = NDManager.newBaseManager()
    // Build a simple linear model: output = input * weights + bias
    val block: Block =
        SequentialBlock()
            .add(Linear.builder().setUnits(nOut).build())

    val model = Model.newInstance("linear-regression")
    model.block = block

    // Prepare mock data
    val inputArr = manager.randomUniform(0f, 1f, Shape(minibatch.toLong(), nIn))
    val labelArr = manager.randomUniform(0f, 1f, Shape(minibatch.toLong(), nOut))
    // Create dataset (not used for iteration, just for shape)
    ArrayDataset
        .Builder()
        .setData(inputArr)
        .optLabels(labelArr)
        .setSampling(minibatch, false)
        .build()

    // Training configuration with MSE loss and Adam optimizer
    val config =
        DefaultTrainingConfig(Loss.l2Loss())
            .optOptimizer(Optimizer.adam().build())

    val trainer = Trainer(model, config)
    trainer.initialize(Shape(minibatch.toLong(), nIn))

    // Forward pass and compute loss
    val batch =
        Batch(
            manager,
            NDList(inputArr),
            NDList(labelArr),
            minibatch,
            Batchifier.STACK,
            Batchifier.STACK,
            0L,
            0L,
        )
    val inputND = batch.data.head()
    println(inputND)
    val labelND = batch.labels.head()

    val numEpochs = 100
    var lastLoss: NDArray? = null

    for (epoch in 1..numEpochs) {
        // Forward pass
        val pred: NDArray
        val lossValue: NDArray

        trainer.newGradientCollector().use { collector ->
            pred = trainer.forward(NDList(inputND)).singletonOrThrow()
            lossValue = trainer.loss.evaluate(NDList(labelND), NDList(pred))
            collector.backward(lossValue)
        }
        trainer.step()
        if (epoch % 10 == 0 || epoch == 1) {
            println("Epoch $epoch, MSE: $lossValue")
        }
        lastLoss = lossValue
    }

//    println("MSE: ${lossValue}")

    // Get gradients for weights and bias
    val params = trainer.model.block.parameters
    println(params.keys())
    val weightGrad =
        params
            .find { it.key == "01Linear_weight" }
            ?.value
            ?.array
            ?.gradient
            ?.duplicate()
    val biasGrad =
        params
            .find { it.key == "01Linear_bias" }
            ?.value
            ?.array
            ?.gradient
            ?.duplicate()

    println("Weights gradient:")
    println(weightGrad)
    println("Bias gradient:")
    println(biasGrad)
}

/**
 * Represents Test1.
 */
class Test1
