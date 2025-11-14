package jp.live.ugai.d2j.attention

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import kotlin.math.exp
import kotlin.math.pow

class Chap10UtilsTest {
    @Test
    fun testBleu() {
        val predSeq = "the cat is on the mat"
        val labelSeq = "the cat is on the mat"
        val bleuScore = Chap10Utils.bleu(predSeq, labelSeq, 4)
        assertEquals(1.0, bleuScore, 1e-6)

        val predSeq2 = "a cat is on the mat"
        val labelSeq2 = "the cat is on the mat"
        val bleuScore2 = Chap10Utils.bleu(predSeq2, labelSeq2, 4)
        val p1 = 5.0 / 6.0
        val p2 = 4.0 / 5.0
        val p3 = 3.0 / 4.0
        val p4 = 2.0 / 3.0
        val expected2 = exp(0.0) * (p1 * p2 * p3 * p4).pow(1.0 / 4.0)
        assertEquals(expected2, bleuScore2, 1e-6)

        val ref = "the cat is on the mat"
        val cand = "the the the cat mat"
        val bleuScore3 = Chap10Utils.bleu(cand, ref, 4)
        assertEquals(0.0, bleuScore3, 1e-6)
    }

    @Test
    fun testMaskedSoftmax() {
        val manager = NDManager.newBaseManager()
        // Input shape: (batch_size=1, num_queries=2, num_keys=3)
        val input = manager.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), Shape(1, 2, 3))
        // Valid lengths shape: (batch_size=1, num_queries=2)
        // For the first query, 1 key is valid. For the second, 3 keys are valid.
        val validLens = manager.create(arrayOf(intArrayOf(1, 3)))
        val result = Chap10Utils.maskedSoftmax(input, validLens)

        // Expected result after masking (before softmax)
        // Query 1: [1, -inf, -inf]
        // Query 2: [4, 5, 6]
        val expectedInput = manager.create(floatArrayOf(1f, -1.0E6f, -1.0E6f, 4f, 5f, 6f), Shape(1, 2, 3))
        val expected = expectedInput.softmax(-1)

        // Compare the results
        assertEquals(expected.toFloatArray().joinToString(), result.toFloatArray().joinToString())
    }

    @Test
    fun testTransposeQkv() {
        val manager = NDManager.newBaseManager()
        val input = manager.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), Shape(2, 2, 2))
        val result = Chap10Utils.transposeQkv(input, 2)
        val expected = manager.create(floatArrayOf(1f, 3f, 2f, 4f, 5f, 7f, 6f, 8f), Shape(4, 1, 2))
        assertEquals(
            expected.toFloatArray().joinToString(),
            result.toFloatArray().joinToString(),
        )
    }

    @Test
    fun testTransposeOutput() {
        val manager = NDManager.newBaseManager()
        val input = manager.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), Shape(4, 1, 2))
        val result = Chap10Utils.transposeOutput(input, 2)
        val expected = manager.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), Shape(2, 2, 2))
        assertEquals(expected.toFloatArray().joinToString(), result.toFloatArray().joinToString())
    }
}
