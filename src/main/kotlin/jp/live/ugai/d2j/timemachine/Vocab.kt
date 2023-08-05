package jp.live.ugai.d2j.timemachine

/**
 * Class representing a vocabulary for a set of tokens.
 *
 * @property tokens List of token lists to be included in the vocabulary.
 * @property minFreq Minimum frequency for a token to be included in the vocabulary.
 * @property reservedTokens List of reserved tokens to be included in the vocabulary regardless of their frequency.
 */
class Vocab(tokens: List<List<String>>, minFreq: Int, reservedTokens: List<String>) {
    /**
     * @property unk Index of the unknown token.
     */
    var unk: Int = 0

    /**
     * @property tokenFreqs List of pairs of tokens and their frequencies.
     */
    var tokenFreqs: List<Pair<String, Int>>

    /**
     * @property idxToToken List of tokens indexed by their indices.
     */
    var idxToToken: MutableList<String> = mutableListOf()

    /**
     * @property tokenToIdx Map of tokens to their indices.
     */
    var tokenToIdx: MutableMap<String, Int> = mutableMapOf()

    init {
        // Sort according to frequencies
        tokenFreqs = countCorpus2D(tokens).toList().sortedByDescending { (_, value) -> value }

        val uniqTokens = mutableListOf("<unk>").apply { addAll(reservedTokens) }
        uniqTokens.addAll(tokenFreqs.filter { it.second >= minFreq && !uniqTokens.contains(it.first) }.map { it.first })

        uniqTokens.forEachIndexed { index, token ->
            idxToToken.add(token)
            tokenToIdx[token] = index
        }
    }

    /**
     * Returns the size of the vocabulary.
     *
     * @return Size of the vocabulary.
     */
    fun length() = idxToToken.size

    /**
     * Returns the indices of the given list of tokens.
     *
     * @param tokens List of tokens.
     * @return List of indices.
     */
    fun getIdxs(tokens: List<String>) = tokens.map { getIdx(it) }

    /**
     * Returns the index of the given token.
     *
     * @param token Token.
     * @return Index of the token.
     */
    fun getIdx(token: String) = tokenToIdx.getOrDefault(token, unk)

    /**
     * Returns the tokens corresponding to the given list of indices.
     *
     * @param indices List of indices.
     * @return List of tokens.
     */
    fun toTokens(indices: List<Int>) = indices.map { toToken(it) }

    /**
     * Returns the token corresponding to the given index.
     *
     * @param index Index.
     * @return Token.
     */
    fun toToken(index: Int) = idxToToken[index]

    companion object {
        /**
         * Counts the frequency of each token in the given list of tokens.
         *
         * @param T Type of the tokens.
         * @param tokens List of tokens.
         * @return Map of tokens to their frequencies.
         */
        fun <T> countCorpus(tokens: List<T>) = tokens.groupingBy { it }.eachCount()

        /**
         * Flattens a list of token lists into a list of tokens and counts the frequency of each token.
         *
         * @param T Type of the tokens.
         * @param tokens List of token lists.
         * @return Map of tokens to their frequencies.
         */
        fun <T> countCorpus2D(tokens: List<List<T>>) = countCorpus(tokens.flatten())
    }
}
