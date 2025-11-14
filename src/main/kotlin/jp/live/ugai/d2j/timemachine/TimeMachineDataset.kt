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

/**
 * Dataset for the Time Machine text, supporting random access and sequence partitioning.
 *
 * Prepares input and label NDArray batches for language modeling tasks. Uses a builder
 * pattern for flexible configuration.
 */
class TimeMachineDataset(
    builder: Builder,
) : RandomAccessDataset(builder) {
    /**
     * Vocabulary built from the corpus, used for tokenization and mapping.
     */
    var vocab: Vocab? = null

    private var data: NDArray // Input data NDArray for model training
    private var labels: NDArray // Label NDArray for next-token prediction
    private val numSteps: Int // Number of time steps in each sequence
    private val maxTokens: Int // Maximum number of tokens to use from the corpus
    private val batchSize: Int // Batch size for training
    private val manager: NDManager // NDManager for NDArray lifecycle management
    private var prepared: Boolean // Flag to indicate if dataset is prepared

    init {
        numSteps = builder.numSteps
        maxTokens = builder.maxTokens
        batchSize = builder.sampler.batchSize
        manager = requireNotNull(builder.manager) { "NDManager must not be null" }
        data = manager.create(Shape(0, 35), DataType.INT32)
        labels = manager.create(Shape(0, 35), DataType.INT32)
        prepared = false
    }

    /**
     * Returns a Record containing the input and label NDArrays for the given index.
     *
     * @param manager NDManager for array management.
     * @param index Index of the sample to retrieve.
     * @return Record containing input and label NDList.
     */
    override fun get(
        manager: NDManager,
        index: Long,
    ): Record {
        val x = data[NDIndex("{}", index)]
        val y = labels[NDIndex("{}", index)]
        return Record(NDList(x), NDList(y))
    }

    /**
     * Returns the number of available samples in the dataset.
     *
     * @return Number of samples as Long.
     */
    override fun availableSize(): Long = data.shape[0]

    /**
     * Prepares the dataset by loading the corpus, partitioning sequences, and creating NDArrays.
     *
     * @param progress Optional progress tracker.
     */
    override fun prepare(progress: Progress?) {
        if (prepared) return

        val (corpus, vocab) = loadCorpusTimeMachine(maxTokens)
        this.vocab = vocab

        // Start with a random offset to partition the sequence for batching
        val offset = Random.nextInt(numSteps)
        val numTokens = ((corpus.size - offset - 1) / batchSize) * batchSize

        val xs =
            manager
                .create(corpus.subList(offset, offset + numTokens).toIntArray())
                .reshape(Shape(batchSize.toLong(), -1))
        val ys =
            manager
                .create(corpus.subList(offset + 1, offset + 1 + numTokens).toIntArray())
                .reshape(Shape(batchSize.toLong(), -1))

        val numBatches = xs.shape[1].toInt() / numSteps
        val xNDList = NDList()
        val yNDList = NDList()

        for (i in 0 until numSteps * numBatches step numSteps) {
            xNDList.add(xs[NDIndex(":, {}:{}", i, i + numSteps)])
            yNDList.add(ys[NDIndex(":, {}:{}", i, i + numSteps)])
        }

        data = NDArrays.concat(xNDList)
        xNDList.close()
        labels = NDArrays.concat(yNDList)
        yNDList.close()
        prepared = true
    }

    /**
     * Builder class for constructing a TimeMachineDataset with custom parameters.
     */
    class Builder : BaseBuilder<Builder>() {
        /**
         * Number of steps (sequence length) for each sample in the dataset.
         */
        var numSteps = 0

        /**
         * Maximum number of tokens to use from the corpus.
         */
        var maxTokens = 0

        /**
         * NDManager instance for managing NDArray lifecycles.
         */
        var manager: NDManager? = null

        /**
         * Returns this Builder instance for method chaining.
         */
        override fun self() = this

        /**
         * Sets the number of steps (sequence length) for the dataset.
         *
         * @param steps Number of steps per sample.
         * @return This Builder instance.
         */
        fun setSteps(steps: Int) = apply { numSteps = steps }

        /**
         * Sets the maximum number of tokens to use from the corpus.
         *
         * @param maxTokens Maximum number of tokens.
         * @return This Builder instance.
         */
        fun setMaxTokens(maxTokens: Int) = apply { this.maxTokens = maxTokens }

        /**
         * Sets the NDManager for NDArray lifecycle management.
         *
         * @param manager NDManager instance.
         * @return This Builder instance.
         */
        fun setManager(manager: NDManager) = apply { this.manager = manager }

        /**
         * Builds and returns a TimeMachineDataset instance with the configured parameters.
         *
         * @return Configured TimeMachineDataset instance.
         */
        fun build() = TimeMachineDataset(this)
    }
}
