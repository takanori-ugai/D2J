package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.SequentialBlock
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class SequentialBlockTest {
    @Test
    fun testOutputShape() {
        val manager = NDManager.newBaseManager()
        val block = SequentialBlock().add(DenseBlock(2, 10))

        val X = manager.randomUniform(0.0f, 1.0f, Shape(4, 3, 8, 8))

        block.initialize(manager, DataType.FLOAT32, X.shape)

        var currentShape = arrayOf(X.shape)
        for (child in block.children.values()) {
            currentShape = child.getOutputShapes(currentShape)
        }

        // Assuming expectedShape is the expected output shape after passing through the block
        val expectedShape = Shape(4, 23, 8, 8) // This is just an example, replace with actual expected shape
        assertEquals(1, currentShape.size)
        assertArrayEquals(expectedShape.shape, currentShape[0].shape)
    }
}
