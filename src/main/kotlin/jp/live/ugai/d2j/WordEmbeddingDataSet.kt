package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager
import ai.djl.util.ZipUtils
import jp.live.ugai.d2j.timemachine.Vocab
import java.io.File
import java.net.URL
import java.nio.file.Paths

fun main() {
    val manager = NDManager.newBaseManager()

    fun readPTB(): List<List<String>> {
        val ptbURL = "http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip"
        val input = URL(ptbURL).openStream()
        ZipUtils.unzip(input, Paths.get("./"))

        val lines = mutableListOf<String>()
        val file = File("./ptb/ptb.train.txt")
        val myReader = java.util.Scanner(file)
        while (myReader.hasNextLine()) {
            lines.add(myReader.nextLine())
        }
        val tokens = mutableListOf<List<String>>()
        for (i in 0 until lines.size) {
            tokens.add(lines.get(i).trim().split(" "))
        }
        return tokens
    }

    val sentences = readPTB()
    println("# sentences: " + sentences.size)

    val vocab = Vocab(sentences, 10, listOf<String>())
    println(vocab.length())

    fun keep(token: String, counter: Map<Any, Double>, numTokens: Int): Boolean {
        // Return True if to keep this token during subsampling
        return kotlin.random.Random.nextFloat() < Math.sqrt(1e-4 / counter.get(token)!! * numTokens)
    }

    fun subSampling(sentences: List<List<String>>, vocab: Vocab): List<List<String>> {
        val tempSentences = mutableListOf<List<String>>()
        for (i in 0 until sentences.size) {
            val tmp = mutableListOf<String>()
            for (j in 0 until sentences[i].size) {
                tmp.add(vocab.idxToToken.get(vocab.getIdx(sentences[i][j])))
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
}

class WordEmbeddingDataSet
