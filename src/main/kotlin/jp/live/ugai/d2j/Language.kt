package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import jp.live.ugai.d2j.timemachine.TimeMachine.readTimeMachine
import jp.live.ugai.d2j.timemachine.TimeMachine.tokenize
import jp.live.ugai.d2j.timemachine.Vocab
import kotlin.random.Random

fun main() {
    val manager = NDManager.newBaseManager()
    val tokens = tokenize(readTimeMachine(), "word")
// Since each text line is not necessarily a sentence or a paragraph, we
// concatenate all text lines
// Since each text line is not necessarily a sentence or a paragraph, we
// concatenate all text lines
    val corpus: MutableList<String> = mutableListOf()
    for (i in tokens.indices) {
        for (j in tokens[i].indices) {
            if (tokens[i][j] !== "") {
                corpus.add(tokens[i][j])
            }
        }
    }

    val vocab = Vocab(listOf(corpus), -1, listOf<String>())
    println(vocab.tokenFreqs.size)
    for (i in 0..9) {
        val (key, value) = vocab.tokenFreqs[i]
        println("$key: $value")
    }

    val bigramTokens = mutableListOf<String>()
    for (i in 0 until corpus.size - 1) {
        bigramTokens.add(corpus[i] + " " + corpus[i + 1])
    }
    val bigramVocab = Vocab(listOf(bigramTokens), -1, listOf<String>())
    for (i in 0..9) {
        val (key, value) = bigramVocab.tokenFreqs[i]
        println("$key: $value")
    }

    val trigramTokens = mutableListOf<String>()
    for (i in 0 until corpus.size - 2) {
        trigramTokens.add(corpus[i] + " " + corpus[i + 1] + " " + corpus[i + 2])
    }
    val trigramVocab = Vocab(listOf(trigramTokens), -1, listOf<String>())
    for (i in 0..9) {
        val (key, value) = trigramVocab.tokenFreqs[i]
        println("$key: $value")
    }

    val mySeq: MutableList<Int> = mutableListOf()
    for (i in 0..34) {
        mySeq.add(i)
    }

    for (pair in seqDataIterRandom(mySeq, 2, 5, manager)) {
        println(
            """
            X:
            ${pair[0].toDebugString(50, 50, 50, 50)}
            """.trimIndent()
        )
        println(
            """
            Y:
            ${pair[1].toDebugString(50, 50, 50, 50)}
            """.trimIndent()
        )
    }

    for (pair in seqDataIterSequential(mySeq, 2, 5, manager)) {
        println(
            """
            X:
            ${pair[0].toDebugString(10, 10, 10, 10)}
            """.trimIndent()
        )
        println(
            """
            Y:
            ${pair[1].toDebugString(10, 10, 10, 10)}
            """.trimIndent()
        )
    }
}

/**
 * Generate a minibatch of subsequences using random sampling.
 */
fun seqDataIterRandom(corpus: List<Int>, batchSize: Int, numSteps: Int, manager: NDManager): List<NDList> {
    // Start with a random offset (inclusive of `numSteps - 1`) to partition a
    // sequence
    var corpus = corpus
    corpus = corpus.subList(Random.nextInt(numSteps - 1), corpus.size)
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
        val initialIndicesPerBatch: List<Int> = initialIndices.subList(i, i + batchSize)
        val xNDArray: NDArray = manager.create(Shape(initialIndices.size.toLong(), numSteps.toLong()), DataType.INT32)
        val yNDArray: NDArray = manager.create(Shape(initialIndices.size.toLong(), numSteps.toLong()), DataType.INT32)
        for (j in initialIndices.indices) {
            val X = data(initialIndices[j], corpus, numSteps)
            xNDArray[NDIndex(j.toLong())] = manager.create(X.toIntArray())
            val Y = data(initialIndices[j] + 1, corpus, numSteps)
            yNDArray[NDIndex(j.toLong())] = manager.create(Y.toIntArray())
        }
        val pair = NDList()
        pair.add(xNDArray)
        pair.add(yNDArray)
        pairs.add(pair)
        i += batchSize
    }
    return pairs
}

fun data(pos: Int, corpus: List<Int>, numSteps: Int): List<Int> {
    // Return a sequence of length `numSteps` starting from `pos`
    return corpus.subList(pos, pos + numSteps)
}

/**
 * Generate a minibatch of subsequences using sequential partitioning.
 */
fun seqDataIterSequential(
    corpus: List<Int>,
    batchSize: Int,
    numSteps: Int,
    manager: NDManager
): List<NDList> {
    // Start with a random offset to partition a sequence
    val offset = Random.nextInt(numSteps)
    val numTokens = (corpus.size - offset - 1) / batchSize * batchSize
    var Xs = manager.create(
        corpus.subList(offset, offset + numTokens).toIntArray()
    )
    var Ys = manager.create(
        corpus.subList(offset + 1, offset + 1 + numTokens).toIntArray()
    )
    Xs = Xs.reshape(Shape(batchSize.toLong(), -1))
    Ys = Ys.reshape(Shape(batchSize.toLong(), -1))
    val numBatches = Xs.shape[1].toInt() / numSteps
    val pairs = mutableListOf<NDList>()
    var i = 0
    while (i < numSteps * numBatches) {
        val X = Xs[NDIndex(":, {}:{}", i, i + numSteps)]
        val Y = Ys[NDIndex(":, {}:{}", i, i + numSteps)]
        val pair = NDList()
        pair.add(X)
        pair.add(Y)
        pairs.add(pair)
        i += numSteps
    }
    return pairs
}

class Language
