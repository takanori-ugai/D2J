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
        val input = manager.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), Shape(2, 3))
        val validLens = manager.create(intArrayOf(1, 2))
        val result = Chap10Utils.maskedSoftmax(input, validLens)

        val expected = manager.create(floatArrayOf(1.0f, -1.0E6f, -1.0E6f, 4f, 5f, -1.0E6f), Shape(2, 3)).softmax(-1)
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
