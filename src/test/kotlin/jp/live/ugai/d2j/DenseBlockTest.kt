package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test

class DenseBlockTest {
    @Test
    fun testTransitionBlock() {
        val manager = NDManager.newBaseManager()
        val block = transitionBlock(10)
        val x = manager.randomUniform(0.0f, 1.0f, Shape(4, 23, 8, 8))
        block.initialize(manager, DataType.FLOAT32, x.shape)
        var currentShape = arrayOf(x.shape)
        for (child in block.children.values()) {
            currentShape = child.getOutputShapes(currentShape)
        }
        val expectedShape = Shape(4, 10, 4, 4)
        assertArrayEquals(expectedShape.shape, currentShape[0].shape)
    }

    @Test
    fun testDenseBlockOutputShape() {
        val manager = NDManager.newBaseManager()
        val block = DenseBlock(2, 10)
        val x = manager.randomUniform(0.0f, 1.0f, Shape(4, 3, 8, 8))
        block.initialize(manager, DataType.FLOAT32, x.shape)
        val currentShape = block.getOutputShapes(arrayOf(x.shape))
        val expectedShape = Shape(4, 23, 8, 8)
        assertArrayEquals(expectedShape.shape, currentShape[0].shape)
    }

    @Test
    fun testDenseBlockForward() {
        val manager = NDManager.newBaseManager()
        val block = DenseBlock(2, 10)
        val x = manager.randomUniform(0.0f, 1.0f, Shape(4, 3, 8, 8))
        block.initialize(manager, DataType.FLOAT32, x.shape)
        val output = block.forward(ai.djl.training.ParameterStore(), ai.djl.ndarray.NDList(x), false).singletonOrThrow()
        val expectedShape = Shape(4, 23, 8, 8)
        assertArrayEquals(expectedShape.shape, output.shape.shape)
    }
}
