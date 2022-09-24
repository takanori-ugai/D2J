package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.util.DownloadUtils
import jp.live.ugai.d2j.timemachine.Vocab
import java.io.File
import java.io.InputStream
import java.nio.charset.StandardCharsets
import java.util.Enumeration
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipFile

fun main() {
    val manager = NDManager.newBaseManager()

    fun readDataNMT(): String? {
        DownloadUtils.download("http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip", "fra-eng.zip")
        val zipFile = ZipFile(File("fra-eng.zip"))
        val entries: Enumeration<out ZipEntry> = zipFile.entries()
        while (entries.hasMoreElements()) {
            val entry: ZipEntry = entries.nextElement()
            if (entry.name.contains("fra.txt")) {
                val stream: InputStream = zipFile.getInputStream(entry)
                return String(stream.readAllBytes(), StandardCharsets.UTF_8)
            }
        }
        return null
    }

    val rawText = readDataNMT()
    println(rawText?.substring(0, 75))

    fun noSpace(currChar: Char, prevChar: Char): Boolean {
        /* Preprocess the English-French dataset. */
        return (listOf(',', '.', '!', '?').contains(currChar) && prevChar != ' ')
    }

    fun preprocessNMT(_text: String): String {
        // Replace non-breaking space with space, and convert uppercase letters to
        // lowercase ones
        var text = _text.replace('\u202f', ' ').replace("\\xa0".toRegex(), " ").lowercase(Locale.getDefault())

        // Insert space between words and punctuation marks
        val out = StringBuilder()
        var currChar: Char
        for (i in 0 until text.length) {
            currChar = text[i]
            if (i > 0 && noSpace(currChar, text[i - 1])) {
                out.append(' ')
            }
            out.append(currChar)
        }
        return out.toString()
    }

    val text = preprocessNMT(rawText!!)
    println(text.substring(0, 80))

    fun tokenizeNMT(text: String, numExamples: Int?): Pair<List<List<String>>, List<List<String>>> {
        val source = mutableListOf<List<String>>()
        val target = mutableListOf<List<String>>()
        var i = 0
        for (line in text.split("\n".toRegex()).dropLastWhile { it.isEmpty() }) {
            if (numExamples != null && i > numExamples) {
                break
            }
            val parts = line.split("\t".toRegex()).dropLastWhile { it.isEmpty() }
            if (parts.size == 2) {
                source.add(parts[0].split(" ".toRegex()).dropLastWhile { it.isEmpty() })
                target.add(parts[1].split(" ".toRegex()).dropLastWhile { it.isEmpty() })
            }
            i += 1
        }
        return Pair(source, target)
    }

    val pair: Pair<List<List<String>>, List<List<String>>> = tokenizeNMT(text, null)
    val source: List<List<String>> = pair.first
    val target: List<List<String>> = pair.second
    for (subArr in source.subList(0, 5)) {
        println(subArr)
    }

    for (subArr in target.subList(0, 5)) {
        println(subArr)
    }

    val y1 = mutableListOf<Int>()
    for (i in source.indices) y1.add(source[i].size)
    val y2 = mutableListOf<Int>()
    for (i in target.indices) y2.add(target[i].size)
    println(y1.size)
    println(y2.size)

    var srcVocab = Vocab(source, 2, listOf("<pad>", "<bos>", "<eos>"))
    println(srcVocab.length())

    fun truncatePad(integerLine: List<Int>, numSteps: Int, paddingToken: Int): List<Int> {
        /* Truncate or pad sequences */
        val line = integerLine.toMutableList()
        if (integerLine.size > numSteps) {
            return line.subList(0, numSteps)
        }
        line.addAll(Array<Int>(numSteps - integerLine.size) { paddingToken })
        return line
    }

    val result: List<Int> = truncatePad(srcVocab.getIdxs(source[0]), 10, srcVocab.getIdx("<pad>"))
    println(result)

    fun buildArrayNMT(lines: List<List<String>>, vocab: Vocab, numSteps: Int): Pair<NDArray, NDArray> {
        /* Transform text sequences of machine translation into minibatches. */
        val linesIntArr: MutableList<List<Int>> = mutableListOf()
        for (strings in lines) {
            linesIntArr.add(vocab.getIdxs(strings))
        }
        for (i in linesIntArr.indices) {
            val temp: MutableList<Int> = linesIntArr[i].toMutableList()
            temp.add(vocab.getIdx("<eos>"))
            linesIntArr[i] = temp
        }
        val manager = NDManager.newBaseManager()
        val arr: NDArray = manager.create(Shape(linesIntArr.size.toLong(), numSteps.toLong()), DataType.INT32)
        var row = 0
        for (line in linesIntArr) {
            val rowArr: NDArray = manager.create(truncatePad(line, numSteps, vocab.getIdx("<pad>")).toIntArray())
            arr[NDIndex("{}:", row)] = rowArr
            row += 1
        }
        val validLen = arr.neq(vocab.getIdx("<pad>")).sum(intArrayOf(1))
        return Pair(arr, validLen)
    }

    fun loadDataNMT(batchSize: Int, numSteps: Int, numExamples: Int): Pair<ArrayDataset, Pair<Vocab, Vocab>> {
        /* Return the iterator and the vocabularies of the translation dataset. */
        val text: String = preprocessNMT(readDataNMT()!!)
        val pair: Pair<List<List<String>>, List<List<String>>> = tokenizeNMT(text, numExamples)
        val source: List<List<String>> = pair.first
        val target: List<List<String>> = pair.second
        val srcVocab = Vocab(source, 2, listOf("<pad>", "<bos>", "<eos>"))
        val tgtVocab = Vocab(target, 2, listOf("<pad>", "<bos>", "<eos>"))
        var pairArr: Pair<NDArray, NDArray> = buildArrayNMT(source, srcVocab, numSteps)
        val srcArr: NDArray = pairArr.first
        val srcValidLen: NDArray = pairArr.second
        pairArr = buildArrayNMT(target, tgtVocab, numSteps)
        val tgtArr: NDArray = pairArr.first
        val tgtValidLen: NDArray = pairArr.second
        val dataset = ArrayDataset.Builder()
            .setData(srcArr, srcValidLen)
            .optLabels(tgtArr, tgtValidLen)
            .setSampling(batchSize, true)
            .build()
        return Pair(dataset, Pair(srcVocab, tgtVocab))
    }

    val output = loadDataNMT(2, 8, 600)
    val dataset: ArrayDataset = output.first
    srcVocab = output.second.first
    val tgtVocab: Vocab = output.second.second

    val batch = dataset.getData(manager).iterator().next()
    val X = batch.data[0]
    val xValidLen = batch.data[1]
    val Y = batch.labels[0]
    val yValidLen = batch.labels[1]
    println(X)
    println(xValidLen)
    println(Y)
    println(yValidLen)
}

class MachineTranslation
