package jp.live.ugai.d2j.timemachine

import java.util.*

class Vocab(tokens: List<List<String>>, minFreq: Int, reservedTokens: List<String>) {
    // The index for the unknown token is 0
    var unk: Int = 0
    var tokenFreqs: List<Pair<String, Int>>
    var idxToToken: MutableList<String> = mutableListOf()
    var tokenToIdx: MutableMap<String, Int> = mutableMapOf()

    init {
        // Sort according to frequencies
        tokenFreqs = countCorpus2D(tokens).toList().sortedByDescending { (_, value) -> value }

        val uniqTokens: MutableList<String> = mutableListOf()
        uniqTokens.add("<unk>")
        uniqTokens.addAll(reservedTokens)
        for ((key, value) in tokenFreqs) {
            if (value >= minFreq && !uniqTokens.contains(key)) {
                uniqTokens.add(key)
            }
        }
        println("UniqTokens: ${uniqTokens.size}")
        for (token in uniqTokens) {
            idxToToken.add(token)
            tokenToIdx[token] = idxToToken.size - 1
        }
    }

    fun length(): Int {
        return idxToToken.size
    }

    fun getIdxs(tokens: List<String>): List<Int> {
        val idxs: MutableList<Int> = mutableListOf()
        for (token in tokens) {
            idxs.add(getIdx(token))
        }
        return idxs
    }

    fun getIdx(token: String): Int {
        return tokenToIdx.getOrDefault(token, unk)
    }

    fun toTokens(indices: List<Int>): List<String> {
        val tokens: MutableList<String> = mutableListOf()
        for (index in indices) {
            tokens.add(toToken(index))
        }
        return tokens
    }

    fun toToken(index: Int): String {
        return idxToToken[index]
    }

    companion object {
        /** Count token frequencies.  */
        fun <T> countCorpus(tokens: List<T>): Map<T, Int> {
            val counter = mutableMapOf<T, Int>()
            for (token in tokens) {
                counter[token] = counter.getOrDefault(token, 0) + 1
            }
            return counter
        }

        /** Flatten a list of token lists into a list of tokens  */
        fun <T> countCorpus2D(tokens: List<List<T>>): Map<T, Int> {
            val allTokens: MutableList<T> = mutableListOf()
            for (token in tokens) {
                for (t in token) {
                    if (t !== "") {
                        allTokens.add(t)
                    }
                }
            }
            return countCorpus(allTokens)
        }
    }
}
