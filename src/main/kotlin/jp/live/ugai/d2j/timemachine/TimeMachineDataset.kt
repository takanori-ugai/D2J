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

    override fun get(manager: NDManager, index: Long): Record {
        val X = data[NDIndex("{}", index)]
        val Y = labels[NDIndex("{}", index)]
        return Record(NDList(X), NDList(Y))
    }

    override fun availableSize(): Long {
        return data.shape[0]
    }

    override fun prepare(progress: Progress?) {
        if (prepared) {
            return
        }
        var corpusVocabPair = loadCorpusTimeMachine(maxTokens)
        val corpus: List<Int> = corpusVocabPair.first
        vocab = corpusVocabPair.second

        // Start with a random offset (inclusive of `numSteps - 1`) to partition a
        // sequence
        val offset: Int = Random.nextInt(numSteps)
        val numTokens = ((corpus.size - offset - 1) / batchSize) * batchSize
        var Xs = manager!!.create(corpus.subList(offset, offset + numTokens).toIntArray())
        var Ys = manager.create(corpus.subList(offset + 1, offset + 1 + numTokens).toIntArray())
        Xs = Xs.reshape(Shape(batchSize.toLong(), -1))
        Ys = Ys.reshape(Shape(batchSize.toLong(), -1))
        val numBatches = Xs.shape[1].toInt() / numSteps
        val xNDList = NDList()
        val yNDList = NDList()
        var i = 0
        while (i < numSteps * numBatches) {
            val X = Xs[NDIndex(":, {}:{}", i, i + numSteps)]
            val Y = Ys[NDIndex(":, {}:{}", i, i + numSteps)]
            xNDList.add(X)
            yNDList.add(Y)
            i += numSteps
        }
        data = NDArrays.concat(xNDList)
        xNDList.close()
        labels = NDArrays.concat(yNDList)
        yNDList.close()
        prepared = true
    }

    class Builder : BaseBuilder<Builder>() {
        var numSteps = 0
        var maxTokens = 0
        var manager: NDManager? = null
        override fun self(): Builder {
            return this
        }

        fun setSteps(steps: Int): Builder {
            numSteps = steps
            return this
        }

        fun setMaxTokens(maxTokens: Int): Builder {
            this.maxTokens = maxTokens
            return this
        }

        fun setManager(manager: NDManager): Builder {
            this.manager = manager
            return this
        }

        fun build(): TimeMachineDataset {
            return TimeMachineDataset(this)
        }
    }
}
