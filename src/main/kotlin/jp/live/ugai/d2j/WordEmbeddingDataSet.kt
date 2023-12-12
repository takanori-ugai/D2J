package jp.live.ugai.d2j

import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.training.dataset.ArrayDataset
import ai.djl.translate.Batchifier
import ai.djl.util.ZipUtils
import jp.live.ugai.d2j.timemachine.Vocab
import org.apache.commons.math3.distribution.EnumeratedDistribution
import org.jetbrains.letsPlot.geom.geomHistogram
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.letsPlot
import java.io.File
import java.net.URL
import java.nio.file.Paths

fun main() {
    val manager = NDManager.newBaseManager()

    fun readPTB(): List<List<String>> {
        val ptbURL = "http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip"
        val input = URL(ptbURL).openStream()
        ZipUtils.unzip(input, Paths.get("./"))
        input.close()
        val lines = mutableListOf<String>()
        val file = File("./ptb/ptb.train.txt")
        val myReader = java.util.Scanner(file)
        while (myReader.hasNextLine()) {
            lines.add(myReader.nextLine())
        }
        val tokens = mutableListOf<List<String>>()
        for (i in lines) {
            tokens.add(i.trim().split(" "))
        }
        myReader.close()
        return tokens
    }

    val sentences = readPTB()
    println("# sentences: " + sentences.size)

    var vocab = Vocab(sentences, 10, listOf<String>())
    println(vocab.length())

    fun keep(
        token: String,
        counter: Map<Any, Double>,
        numTokens: Int,
    ): Boolean {
        // Return True if to keep this token during subsampling
        return kotlin.random.Random.nextFloat() < Math.sqrt(1e-4 / counter[token]!! * numTokens)
    }

    fun subSampling(
        sentences: List<List<String>>,
        vocab: Vocab,
    ): List<List<String>> {
        val tempSentences = mutableListOf<List<String>>()
        for (i in sentences.indices) {
            val tmp = mutableListOf<String>()
            for (j in 0 until sentences[i].size) {
                tmp.add(vocab.idxToToken[vocab.getIdx(sentences[i][j])])
            }
            tempSentences.add(tmp)
        }
        // Count the frequency for each word
        val counter = Vocab.countCorpus2D(sentences)
        var numTokens: Int = 0
        for (value in counter.values) {
            numTokens += value
        }

        // Now do the subsampling
        val output = mutableListOf<List<String>>()
        for (i in 0 until tempSentences.size) {
            val tks = mutableListOf<String>()
            for (j in 0 until tempSentences[i].size) {
                val tk = tempSentences[i][j]
                if (keep(tempSentences[i][j], counter.map { it.key to it.value.toDouble() }.toMap(), numTokens)) {
                    tks.add(tk)
                }
            }
            output.add(tks)
        }

        return output
    }

    val subsampled = subSampling(sentences, vocab)

    val y1 = sentences.map { it.size }
    val y2 = subsampled.map { it.size }

    val label = List<String>(sentences.size) { "body" }
    val label2 = List<String>(subsampled.size) { "subsampled" }
    val data = mapOf("x" to y1 + y2, "cond" to label + label2)
    var plot =
        letsPlot(data) {
            x = "x"
            fill = "cond"
        }
    plot += geomHistogram(binWidth = 5)
    plot += ggsize(500, 250)

    fun compareCounts(
        token: String,
        sentences: List<List<String>>,
        subsampled: List<List<String>>,
    ): String {
        var beforeCount = 0
        for (element in sentences) {
            beforeCount += element.count { it.equals(token) }
        }

        var afterCount = 0
        for (element in subsampled) {
            afterCount += element.count { it.equals(token) }
        }

        return "# of \"the\": before=$beforeCount, after=$afterCount"
    }

    println(compareCounts("the", sentences, subsampled))

    println(compareCounts("join", sentences, subsampled))

    val corpus = mutableListOf<List<Int>>()
    for (element in subsampled) {
        corpus.add(vocab.getIdxs(element))
    }
    for (i in 0 until 3) {
        println(corpus[i])
    }

    fun getCentersAndContext(
        corpus: List<List<Int>>,
        maxWindowSize: Int,
    ): kotlin.Pair<List<Int>, List<List<Int>>> {
        var centers = mutableListOf<Int>()
        val contexts = mutableListOf<List<Int>>()

        for (line in corpus) {
            // Each sentence needs at least 2 words to form a "central target word
            // - context word" pair
            if (line.size < 2) {
                continue
            }
            centers.addAll(line)
            for (i in line.indices) { // Context window centered at i
                val windowSize = kotlin.random.Random.nextInt(maxWindowSize - 1) + 1
                val indices =
                    (Math.max(0, i - windowSize) until Math.min(line.size, i + 1 + windowSize)).toMutableList()
                // Exclude the central target word from the context words
                indices.remove(indices.indexOf(i))
                val context = indices.map { line[it] }
                contexts.add(context)
            }
        }
        return kotlin.Pair(centers, contexts)
    }

    val tinyDataset =
        listOf<List<Int>>(
            (0 until 7).toList(),
            (7 until 10).toList(),
        )

    println("dataset $tinyDataset")
    var centerContextPair = getCentersAndContext(tinyDataset, 2)
    println(centerContextPair)
    for (i in 0 until centerContextPair.second.size) {
        println(
            "Center " +
                centerContextPair.first.get(i) +
                " has contexts" +
                centerContextPair.second.get(i),
        )
    }

    centerContextPair = getCentersAndContext(corpus, 5)
    val allCenters = centerContextPair.first
    val allContexts = centerContextPair.second
    println("# center-context pairs:${allCenters.size}")

    class RandomGenerator(samplingWeights: List<Double>) {
        // Draw a random int in [0, n] according to n sampling weights.

        private val population: kotlin.collections.List<Int>
        private val samplingWeights: kotlin.collections.List<Double>
        private var candidates: MutableList<Int>
        private val pmf: MutableList<org.apache.commons.math3.util.Pair<Int, Double>>
        private var i: Int

        init {
            this.population = List<Int>(samplingWeights.size) { it }
            this.samplingWeights = samplingWeights
            this.candidates = mutableListOf()
            this.i = 0

            this.pmf = mutableListOf()
            for (i in samplingWeights.indices) {
                this.pmf.add(org.apache.commons.math3.util.Pair(this.population.get(i), this.samplingWeights.get(i)))
            }
        }

        fun draw(): Int {
            if (this.i == this.candidates.size) {
                this.candidates =
                    EnumeratedDistribution(this.pmf).sample(10000, arrayOf<Int>()).toMutableList()
                this.i = 0
            }
            this.i += 1
            return this.candidates.get(this.i - 1)
        }
    }

    val generator = RandomGenerator(listOf(2.0, 3.0, 4.0))
    val generatorOutput = List(10) { generator.draw() }
    println(generatorOutput)

    fun getNegatives(
        allContexts: List<List<Int>>,
        corpus: List<List<Int>>,
        K: Int,
    ): List<List<Int>> {
        val counter = Vocab.countCorpus2D(corpus)
        val samplingWeights = mutableListOf<Double>()
        for (entry in counter) {
            samplingWeights.add(Math.pow(entry.value.toDouble(), 0.75))
        }
        val allNegatives = mutableListOf<List<Int>>()
        val generator = RandomGenerator(samplingWeights)
        for (contexts in allContexts) {
            val negatives = mutableListOf<Int>()
            while (negatives.size < contexts.size * K) {
                val neg = generator.draw()
                // Noise words cannot be context words
                if (!contexts.contains(neg)) {
                    negatives.add(neg)
                }
            }
            allNegatives.add(negatives)
        }
        return allNegatives
    }

    val allNegatives = getNegatives(allContexts, corpus, 5)

    fun batchifyData(data: List<NDList>): NDList {
        val centers = NDList()
        val contextsNegatives = NDList()
        val masks = NDList()
        val labels = NDList()

        var maxLen = 0L
        for (ndList in data) { // center, context, negative = ndList
            maxLen =
                Math.max(
                    maxLen,
                    ndList.get(1).countNonzero().getLong() +
                        ndList.get(2).countNonzero().getLong(),
                )
        }
        for (ndList in data) { // center, context, negative = ndList
            val center = ndList.get(0)
            val context = ndList.get(1)
            val negative = ndList.get(2)

            var count = 0L
            for (i in 0 until context.size()) {
                // If a 0 is found, we want to stop adding these
                // values to NDArray
                if (context.get(i).getInt() == 0) {
                    break
                }
                contextsNegatives.add(context.get(i).reshape(1))
                masks.add(manager.create(1).reshape(1))
                labels.add(manager.create(1).reshape(1))
                count += 1
            }
            for (i in 0 until negative.size()) {
                // If a 0 is found, we want to stop adding these
                // values to NDArray
                if (negative.get(i).getInt() == 0) {
                    break
                }
                contextsNegatives.add(negative.get(i).reshape(1))
                masks.add(manager.create(1).reshape(1))
                labels.add(manager.create(0).reshape(1))
                count += 1
            }
            // Fill with zeroes remaining array
            while (count != maxLen) {
                contextsNegatives.add(manager.create(0).reshape(1))
                masks.add(manager.create(0).reshape(1))
                labels.add(manager.create(0).reshape(1))
                count += 1
            }

            // Add this NDArrays to output NDArrays
            centers.add(center.reshape(1))
        }
        return NDList(
            NDArrays.concat(centers).reshape(data.size.toLong(), -1),
            NDArrays.concat(contextsNegatives).reshape(data.size.toLong(), -1),
            NDArrays.concat(masks).reshape(data.size.toLong(), -1),
            NDArrays.concat(labels).reshape(data.size.toLong(), -1),
        )
    }

    val x1 =
        NDList(
            manager.create(intArrayOf(1)),
            manager.create(intArrayOf(2, 2)),
            manager.create(intArrayOf(3, 3, 3, 3)),
        )
    val x2 =
        NDList(
            manager.create(intArrayOf(1)),
            manager.create(intArrayOf(2, 2, 2)),
            manager.create(intArrayOf(3, 3)),
        )

    val batchedData = batchifyData(listOf<NDList>(x1, x2))
    val names = listOf("centers", "contexts_negatives", "masks", "labels")
    for (i in 0 until batchedData.size) {
        println(names[i] + " shape: " + batchedData.get(i))
    }

    fun convertNDArray(
        data: List<List<Any>>,
        manager: NDManager,
    ): NDList {
        val centers: MutableList<Int> = (data[0] as List<Int>).toMutableList()
        val contexts: List<MutableList<Int>> = (data[1] as List<List<Int>>).map { it.toMutableList() }
        val negatives: List<MutableList<Int>> = (data[2] as List<List<Int>>).map { it.toMutableList() }

        // Create centers NDArray
        val centersNDArray = manager.create(centers.toIntArray())

        // Create contexts NDArray
        var maxLen = 0
        for (context in contexts) {
            maxLen = Math.max(maxLen, context.size)
        }
        // Fill arrays with 0s to all have same lengths and be able to create NDArray
        for (context in contexts) {
            while (context.size != maxLen) {
                context.add(0)
            }
        }
        val contextsNDArray = manager.create(contexts.map { it.toIntArray() }.toTypedArray())

        // Create negatives NDArray
        maxLen = 0
        for (negative in negatives) {
            maxLen = Math.max(maxLen, negative.size)
        }
        // Fill arrays with 0s to all have same lengths and be able to create NDArray
        for (negative in negatives) {
            while (negative.size != maxLen) {
                negative.add(0)
            }
        }
        val negativesNDArray =
            manager.create(
                negatives.map { it.toIntArray() }.toTypedArray(),
            )

        return NDList(centersNDArray, contextsNDArray, negativesNDArray)
    }

    fun loadDataPTB(
        batchSize: Int,
        maxWindowSize: Int,
        numNoiseWords: Int,
        manager: NDManager,
    ): Pair<ArrayDataset, Vocab> {
        val sentences = readPTB()
        val vocab = Vocab(sentences, 10, listOf<String>())
        val subSampled = subSampling(sentences, vocab)
        val corpus = mutableListOf<List<Int>>()
        for (i in 0 until subSampled.size) {
            corpus.add(vocab.getIdxs(subSampled[i]))
        }
        val pair = getCentersAndContext(corpus, maxWindowSize)
        val negatives = getNegatives(pair.second, corpus, numNoiseWords)

        val ndArrays =
            convertNDArray(listOf(pair.first, pair.second, negatives), manager)
        val dataset =
            ArrayDataset.Builder()
                .setData(ndArrays.get(0), ndArrays.get(1), ndArrays.get(2))
                .optDataBatchifier(
                    object : Batchifier {
                        override fun batchify(ndLists: Array<NDList>): NDList {
                            return batchifyData(ndLists.toList())
                        }

                        override fun unbatchify(ndList: NDList): Array<NDList> {
                            return arrayOf<NDList>()
                        }
                    },
                )
                .setSampling(batchSize, true)
                .build()

        return Pair(dataset, vocab)
    }

    val datasetVocab = loadDataPTB(512, 5, 5, manager)
    val dataset = datasetVocab.first
    vocab = datasetVocab.second

    val batch = dataset.getData(manager).iterator().next()
    for (i in 0 until batch.data.size) {
        println(names[i] + " shape: " + batch.data[i].shape)
    }
}

class WordEmbeddingDataSet
