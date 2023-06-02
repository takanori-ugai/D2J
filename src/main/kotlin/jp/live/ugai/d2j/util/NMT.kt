package jp.live.ugai.d2j.util

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.util.DownloadUtils
import jp.live.ugai.d2j.timemachine.Vocab
import java.io.File
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.zip.ZipFile

object NMT {
    fun readDataNMT(): String? {
        DownloadUtils.download(
            "http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip",
            "fra-eng.zip"
        )
        val zipFile = ZipFile(File("fra-eng.zip"))
        val entries = zipFile.entries()
        while (entries.hasMoreElements()) {
            val entry = entries.nextElement()
            if (entry.getName().contains("fra.txt")) {
                zipFile.getInputStream(entry).use { stream ->
                    return String(stream.readAllBytes(), StandardCharsets.UTF_8)
                }
            }
        }
        return null
    }

    fun noSpace(currChar: Char, prevChar: Char): Boolean {
        /* Preprocess the English-French dataset. */
        return listOf(',', '.', '!', '?').contains(currChar) &&
            prevChar != ' '
    }

    fun preprocessNMT(_text: String): String {
        // Replace non-breaking space with space, and convert uppercase letters to
        // lowercase ones

        val text = _text.replace('\u202f', ' ').replace("\\xa0".toRegex(), " ").lowercase(Locale.getDefault())

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

    fun tokenizeNMT(
        text: String,
        numExamples: Int?
    ): Pair<List<List<String>>, List<List<String>>> {
        val source = mutableListOf<List<String>>()
        val target = mutableListOf<List<String>>()

        var i = 0
        for (line in text.split("\n")) {
            if (numExamples != null && i > numExamples) {
                break
            }
            val parts = line.split("\t")
            if (parts.size == 2) {
                source.add(parts[0].split(" "))
                target.add(parts[1].split(" "))
            }
            i += 1
        }
        return Pair(source, target)
    }

    fun truncatePad(integerLine: List<Int>, numSteps: Int, paddingToken: Int): List<Int> {
        /* Truncate or pad sequences */
        val line = integerLine
        if (line.size > numSteps) {
            return line.subList(0, numSteps)
        }
        val paddingTokenArr = List<Int>(numSteps - line.size) { paddingToken } // Pad
        return line + paddingTokenArr
    }

    fun buildArrayNMT(lines: List<List<String>>, vocab: Vocab, numSteps: Int): Pair<NDArray, NDArray> {
        /* Transform text sequences of machine translation into minibatches. */
        val linesIntArr = lines.map { vocab.getIdxs(it) }.toMutableList()
        for (i in linesIntArr.indices) {
            val temp: MutableList<Int> = linesIntArr[i].toMutableList()
            temp.add(vocab.getIdx("<eos>"))
            linesIntArr[i] = temp
        }

        val manager = NDManager.newBaseManager()

        val arr = manager.create(Shape(linesIntArr.size.toLong(), numSteps.toLong()), DataType.INT32)
        var row = 0
        for (line in linesIntArr) {
            val rowArr = manager.create(truncatePad(line, numSteps, vocab.getIdx("<pad>")).toIntArray())
            arr.set(NDIndex("{}:", row), rowArr)
            row += 1
        }
        val validLen = arr.neq(vocab.getIdx("<pad>")).sum(intArrayOf(1))
        return Pair(arr, validLen)
    }

    fun loadDataNMT(
        batchSize: Int,
        numSteps: Int,
        numExamples: Int
    ): Pair<ArrayDataset, Pair<Vocab, Vocab>> {
        /* Return the iterator and the vocabularies of the translation dataset. */
        val text = preprocessNMT(readDataNMT()!!)
        val pair = tokenizeNMT(text, numExamples)
        val source = pair.first
        val target = pair.second
        val srcVocab =
            Vocab(source, 2, listOf("<pad>", "<bos>", "<eos>"))
        val tgtVocab = Vocab(target, 2, listOf("<pad>", "<bos>", "<eos>"))

        var pairArr = buildArrayNMT(source, srcVocab, numSteps)
        val srcArr = pairArr.first
        val srcValidLen = pairArr.second

        pairArr = buildArrayNMT(target, tgtVocab, numSteps)
        val tgtArr = pairArr.first
        val tgtValidLen = pairArr.second

        val dataset = ArrayDataset.Builder()
            .setData(srcArr, srcValidLen)
            .optLabels(tgtArr, tgtValidLen)
            .setSampling(batchSize, true)
            .build()

        return Pair(dataset, Pair(srcVocab, tgtVocab))
    }
}
