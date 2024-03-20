package jp.live.ugai.d2j.timemachine

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.RandomAccessDataset
import ai.djl.training.dataset.Record
import ai.djl.util.Progress
import jp.live.ugai.d2j.timemachine.TimeMachine.loadCorpusTimeMachine
import kotlin.random.Random

class TimeMachineDataset(builder: Builder) : RandomAccessDataset(builder) {
    var vocab: Vocab? = null
    private var data: NDArray
    private var labels: NDArray
    private val numSteps: Int
    private val maxTokens: Int
    private val batchSize: Int
    private val manager: NDManager?
    private var prepared: Boolean

    init {
        numSteps = builder.numSteps
        maxTokens = builder.maxTokens
        batchSize = builder.sampler.batchSize
        manager = builder.manager
        data = manager!!.create(Shape(0, 35), DataType.INT32)
        labels = manager.create(Shape(0, 35), DataType.INT32)
        prepared = false
    }

    override fun get(
        manager: NDManager,
        index: Long,
    ): Record {
        val X = data[NDIndex("{}", index)]
        val Y = labels[NDIndex("{}", index)]
        return Record(NDList(X), NDList(Y))
    }

    override fun availableSize(): Long {
        return data.shape[0]
    }

    override fun prepare(progress: Progress?) {
        if (prepared) return

        val (corpus, vocab) = loadCorpusTimeMachine(maxTokens)
        this.vocab = vocab

        // Start with a random offset (inclusive of `numSteps - 1`) to partition a
        // sequence
        val offset: Int = Random.nextInt(numSteps)
        val numTokens = ((corpus.size - offset - 1) / batchSize) * batchSize
        var Xs = manager!!.create(corpus.subList(offset, offset + numTokens).toIntArray()).reshape(Shape(batchSize.toLong(), -1))
        var Ys = manager.create(
            corpus.subList(offset + 1, offset + 1 + numTokens).toIntArray()
        ).reshape(
            Shape(batchSize.toLong(), -1)
        )
        val numBatches = Xs.shape[1].toInt() / numSteps
        val xNDList = NDList()
        val yNDList = NDList()
        for (i in 0 until numSteps * numBatches step numSteps) {
            xNDList.add(Xs[NDIndex(":, {}:{}", i, i + numSteps)])
            yNDList.add(Ys[NDIndex(":, {}:{}", i, i + numSteps)])
        }
        data = NDArrays.concat(xNDList)
        xNDList.close()
        labels = NDArrays.concat(yNDList)
        yNDList.close()
        prepared = true
    }

    /**
     * Builder class for TimeMachineDataset.
     */
    class Builder : BaseBuilder<Builder>() {
        /**
         * @property numSteps Number of steps to be used in the dataset.
         */
        var numSteps = 0

        /**
         * @property maxTokens Maximum number of tokens to be used in the dataset.
         */
        var maxTokens = 0

        /**
         * @property manager NDManager instance to manage the lifecycle of NDArray.
         */
        var manager: NDManager? = null

        /**
         * Returns the current Builder instance.
         * @return this Builder
         */
        override fun self() = this

        /**
         * Sets the number of steps to be used in the dataset.
         * @param steps Number of steps
         * @return this Builder
         */
        fun setSteps(steps: Int) = apply { numSteps = steps }

        /**
         * Sets the maximum number of tokens to be used in the dataset.
         * @param maxTokens Maximum number of tokens
         * @return this Builder
         */
        fun setMaxTokens(maxTokens: Int) = apply { this.maxTokens = maxTokens }

        /**
         * Sets the NDManager instance to manage the lifecycle of NDArray.
         * @param manager NDManager instance
         * @return this Builder
         */
        fun setManager(manager: NDManager) = apply { this.manager = manager }

        /**
         * Builds a TimeMachineDataset instance with the set parameters.
         * @return TimeMachineDataset instance
         */
        fun build() = TimeMachineDataset(this)
    }
}
