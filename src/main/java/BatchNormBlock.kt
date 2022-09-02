import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.util.PairList

class BatchNormBlock(numFeatures: Int, numDimensions: Int) : AbstractBlock() {

    private var movingMean: NDArray
    private var movingVar: NDArray
    private var gamma: Parameter
    private var beta: Parameter
    private var shape: Shape

    // num_features: the number of outputs for a fully-connected layer
    // or the number of output channels for a convolutional layer.
    // num_dims: 2 for a fully-connected layer and 4 for a convolutional layer.
    init {
        if (numDimensions == 2) {
            shape = Shape(1, numFeatures.toLong())
        } else {
            shape = Shape(1, numFeatures.toLong(), 1, 1)
        }
        // The scale parameter and the shift parameter involved in gradient
        // finding and iteration are initialized to 0 and 1 respectively
        gamma = addParameter(
            Parameter.builder()
                .setName("gamma")
                .setType(Parameter.Type.GAMMA)
                .optShape(shape)
                .build()
        )
        beta = addParameter(
            Parameter.builder()
                .setName("beta")
                .setType(Parameter.Type.BETA)
                .optShape(shape)
                .build()
        )

        // All the variables not involved in gradient finding and iteration are
        // initialized to 0. Create a base manager to maintain their values
        // throughout the entire training process
        val manager = NDManager.newBaseManager()
        movingMean = manager.zeros(shape)
        movingVar = manager.zeros(shape)
    }

    fun batchNormUpdate(
        X: NDArray,
        gamma: NDArray,
        beta: NDArray,
        movingMean0: NDArray,
        movingVar0: NDArray,
        eps: Float,
        momentum: Float,
        isTraining: Boolean
    ): NDList {
        // attach moving mean and var to submanager to close intermediate computation values
        // at the end to avoid memory leak
        var movingMean = movingMean0
        var movingVar = movingVar0
        movingMean.manager.newSubManager().use { subManager ->
            movingMean.attach(subManager)
            movingVar.attach(subManager)
            val xHat: NDArray
            val mean: NDArray
            val vari: NDArray
            if (!isTraining) {
                // If it is the prediction mode, directly use the mean and variance
                // obtained from the incoming moving average
                xHat = X.sub(movingMean).div(movingVar.add(eps).sqrt())
            } else {
                if (X.shape.dimension() == 2) {
                    // When using a fully connected layer, calculate the mean and
                    // variance on the feature dimension
                    mean = X.mean(intArrayOf(0), true)
                    vari = X.sub(mean).pow(2).mean(intArrayOf(0), true)
                } else {
                    // When using a two-dimensional convolutional layer, calculate the
                    // mean and variance on the channel dimension (axis=1). Here we
                    // need to maintain the shape of `X`, so that the broadcast
                    // operation can be carried out later
                    mean = X.mean(intArrayOf(0, 2, 3), true)
                    vari = X.sub(mean).pow(2).mean(intArrayOf(0, 2, 3), true)
                }
                // In training mode, the current mean and variance are used for the
                // standardization
                xHat = X.sub(mean).div(vari.add(eps).sqrt())
                // Update the mean and variance of the moving average
                movingMean = movingMean.mul(momentum).add(mean.mul(1.0f - momentum))
                movingVar = movingVar.mul(momentum).add(vari.mul(1.0f - momentum))
            }
            val Y = xHat.mul(gamma).add(beta) // Scale and shift
            // attach moving mean and var back to original manager to keep their values
            movingMean.attach(subManager.parentManager)
            movingVar.attach(subManager.parentManager)
            return NDList(Y, movingMean, movingVar)
        }
    }

    override fun toString(): String {
        return "BatchNormBlock()"
    }

    override fun forwardInternal(
        parameterStore: ParameterStore?,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>
    ): NDList {
        val result = batchNormUpdate(
            inputs.singletonOrThrow(),
            gamma.array,
            beta.array,
            movingMean,
            movingVar,
            1e-12f,
            0.9f,
            training
        )
        // close previous NDArray before assigning new values
        if (training) {
            movingMean.close()
            movingVar.close()
        }
        // Save the updated `movingMean` and `movingVar`
        movingMean = result[1]
        movingVar = result[2]
        return NDList(result[0])
    }

    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        var current = inputs
        for (block in children.values()) {
            current = block.getOutputShapes(current)
        }
        return current
    }
}
