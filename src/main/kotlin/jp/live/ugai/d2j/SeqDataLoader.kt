package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import jp.live.ugai.d2j.timemachine.TimeMachine.loadCorpusTimeMachine
import jp.live.ugai.d2j.timemachine.Vocab
import java.util.Random

/**
 * Represents SeqDataLoader.
 */
class SeqDataLoader(
    batchSize: Int,
    numSteps: Int,
    useRandomIter: Boolean,
    maxTokens: Int,
) : Iterable<NDList> {
    /**
     * The dataIter.
     */
    var dataIter: List<NDList>

    /**
     * The corpus.
     */
    var corpus: List<Int>

    /**
     * The vocab.
     */
    var vocab: Vocab

    /**
     * The batchSize.
     */
    var batchSize: Int

    /**
     * The numSteps.
     */
    var numSteps: Int

    init {
        /**
         * The manager.
         */
        val manager = NDManager.newBaseManager()

        /**
         * The corpusVocabPair.
         */
        val corpusVocabPair = loadCorpusTimeMachine(maxTokens)
        corpus = corpusVocabPair.first
        vocab = corpusVocabPair.second
        this.batchSize = batchSize
        this.numSteps = numSteps
        if (useRandomIter) {
            dataIter = seqDataIterRandom(corpus, batchSize, numSteps, manager)
        } else {
            dataIter = seqDataIterSequential(corpus, batchSize, numSteps, manager)
        }
    }

    /**
     * Executes iterator.
     */
    override fun iterator(): Iterator<NDList> = dataIter.iterator()

    /**
     * Generate a minibatch of subsequences using random sampling.
     */
    fun seqDataIterRandom(
        corpus: List<Int>,
        batchSize: Int,
        numSteps: Int,
        manager: NDManager,
    ): List<NDList> {
        // Start with a random offset (inclusive of `numSteps - 1`) to partition a
        // sequence
        var corpus = corpus
        corpus = corpus.subList(Random().nextInt(numSteps - 1), corpus.size)
        // Subtract 1 since we need to account for labels
        val numSubseqs = (corpus.size - 1) / numSteps
        // The starting indices for subsequences of length `numSteps`
        val initialIndices: MutableList<Int> = mutableListOf()
        run {
            var i = 0
            while (i < numSubseqs * numSteps) {
                initialIndices.add(i)
                i += numSteps
            }
        }
        // In random sampling, the subsequences from two adjacent random
        // minibatches during iteration are not necessarily adjacent on the
        // original sequence
        initialIndices.shuffle()
        val numBatches = numSubseqs / batchSize
        val pairs = mutableListOf<NDList>()
        var i = 0
        while (i < batchSize * numBatches) {
            // Here, `initialIndices` contains randomized starting indices for
            // subsequences
//            val initialIndicesPerBatch: List<Int> = initialIndices.subList(i, i + batchSize)
            val xNDArray: NDArray =
                manager.create(Shape(initialIndices.size.toLong(), numSteps.toLong()), DataType.INT32)
            val yNDArray: NDArray =
                manager.create(Shape(initialIndices.size.toLong(), numSteps.toLong()), DataType.INT32)
            for (j in initialIndices.indices) {
                val inputSeq = data(initialIndices[j], corpus, numSteps)
                xNDArray[NDIndex(j.toLong())] = manager.create(inputSeq.toIntArray())
                val targetSeq = data(initialIndices[j] + 1, corpus, numSteps)
                yNDArray[NDIndex(j.toLong())] = manager.create(targetSeq.toIntArray())
            }
            val pair = NDList()
            pair.add(xNDArray)
            pair.add(yNDArray)
            pairs.add(pair)
            i += batchSize
        }
        return pairs
    }

    /**
     * Generate a minibatch of subsequences using sequential partitioning.
     */
    fun seqDataIterSequential(
        corpus: List<Int>,
        batchSize: Int,
        numSteps: Int,
        manager: NDManager,
    ): List<NDList> {
        // Start with a random offset to partition a sequence
        val offset = Random().nextInt(numSteps)
        val numTokens = (corpus.size - offset - 1) / batchSize * batchSize
        var inputs =
            manager.create(
                corpus.subList(offset, offset + numTokens).toIntArray(),
            )
        var labels =
            manager.create(
                corpus.subList(offset + 1, offset + 1 + numTokens).toIntArray(),
            )
        inputs = inputs.reshape(Shape(batchSize.toLong(), -1))
        labels = labels.reshape(Shape(batchSize.toLong(), -1))
        val numBatches = inputs.shape[1].toInt() / numSteps
        val pairs = mutableListOf<NDList>()
        var i = 0
        while (i < numSteps * numBatches) {
            val inputSeq = inputs[NDIndex(":, {}:{}", i, i + numSteps)]
            val targetSeq = labels[NDIndex(":, {}:{}", i, i + numSteps)]
            val pair = NDList()
            pair.add(inputSeq)
            pair.add(targetSeq)
            pairs.add(pair)
            i += numSteps
        }
        return pairs
    }

    companion object {
        /**
         * Return the iterator and the vocabulary of the time machine dataset.
         */
        fun loadDataTimeMachine(
            batchSize: Int,
            numSteps: Int,
            useRandomIter: Boolean,
            maxTokens: Int,
        ): Pair<List<NDList>, Vocab> {
            val seqData = SeqDataLoader(batchSize, numSteps, useRandomIter, maxTokens)
            return Pair(seqData.dataIter, seqData.vocab) // ArrayList<NDList>, Vocab
        }
    }
}
