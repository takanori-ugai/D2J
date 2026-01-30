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

/**
 * Utilities for loading and preprocessing the English–French translation dataset.
 */
object NMT {
    /**
     * Downloads and reads the raw English–French dataset text.
     *
     * @return The raw dataset text, or null if the file entry is not found.
     */
    fun readDataNMT(): String? {
        DownloadUtils.download(
            "http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip",
            "fra-eng.zip",
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

    /**
     * Returns true when punctuation should be preceded by a space.
     *
     * @param currChar The current character being examined.
     * @param prevChar The previous character in the string.
     * @return True if a space should be inserted before the punctuation.
     */
    fun noSpace(
        currChar: Char,
        prevChar: Char,
    ): Boolean {
        // Preprocess the English-French dataset.
        return listOf(',', '.', '!', '?').contains(currChar) &&
            prevChar != ' '
    }

    /**
     * Normalizes the raw text by lowercasing and spacing punctuation.
     *
     * @param rawText The raw dataset text.
     * @return The normalized text.
     */
    fun preprocessNMT(rawText: String): String {
        // Replace non-breaking space with space, and convert uppercase letters to
        // lowercase ones

        val text = rawText.replace('\u202f', ' ').replace("\\xa0".toRegex(), " ").lowercase(Locale.getDefault())

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

    /**
     * Tokenizes the dataset into source/target sentence pairs.
     *
     * @param text The normalized dataset text.
     * @param numExamples The maximum number of examples to return, or null for all.
     * @return A pair of tokenized source and target sentence lists.
     */
    fun tokenizeNMT(
        text: String,
        numExamples: Int?,
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

    /**
     * Truncates or pads a token id sequence to a fixed length.
     *
     * @param integerLine The token ids.
     * @param numSteps The target sequence length.
     * @param paddingToken The padding token id.
     * @return The resized sequence.
     */
    fun truncatePad(
        integerLine: List<Int>,
        numSteps: Int,
        paddingToken: Int,
    ): List<Int> {
        // Truncate or pad sequences
        val line = integerLine
        if (line.size > numSteps) {
            return line.subList(0, numSteps)
        }
        val paddingTokenArr = List<Int>(numSteps - line.size) { paddingToken } // Pad
        return line + paddingTokenArr
    }

    /**
     * Converts tokenized sentences into NDArray batches with valid-lengths.
     *
     * @param lines Tokenized sentences.
     * @param vocab Vocabulary used for token ids.
     * @param numSteps The fixed sequence length.
     * @return A pair of (padded token array, valid-length array).
     */
    fun buildArrayNMT(
        lines: List<List<String>>,
        vocab: Vocab,
        numSteps: Int,
    ): Pair<NDArray, NDArray> {
        // Transform text sequences of machine translation into minibatches.
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

    /**
     * Loads the translation dataset as a batched ArrayDataset and vocabularies.
     *
     * @param batchSize The batch size.
     * @param numSteps The fixed sequence length.
     * @param numExamples The number of examples to load.
     * @return The dataset and the source/target vocabularies.
     */
    fun loadDataNMT(
        batchSize: Int,
        numSteps: Int,
        numExamples: Int,
    ): Pair<ArrayDataset, Pair<Vocab, Vocab>> {
        // Return the iterator and the vocabularies of the translation dataset.
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

        val dataset =
            ArrayDataset
                .Builder()
                .setData(srcArr, srcValidLen)
                .optLabels(tgtArr, tgtValidLen)
                .setSampling(batchSize, true)
                .build()

        return Pair(dataset, Pair(srcVocab, tgtVocab))
    }
}
