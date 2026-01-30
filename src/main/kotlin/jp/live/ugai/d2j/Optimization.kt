package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.geom.geomPoint
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.letsPlot

/**
 * Demonstrates the difference between expected risk and empirical risk functions.
 * Computes and prints sample values of both risk functions over a range of x values.
 */
fun main() {
    // Define the target functions
    val expectedRisk: (Float) -> Float = { x -> x * Math.cos(Math.PI * x).toFloat() }
    val empiricalRisk: (Float) -> Float = { x -> expectedRisk(x) + 0.2f * Math.cos(5 * Math.PI * x).toFloat() }

    val manager = NDManager.newBaseManager()
    val xValues = manager.arange(0.5f, 1.5f, 0.01f).toFloatArray()
    val expectedRiskValues = xValues.map(expectedRisk)
    val empiricalRiskValues = xValues.map(empiricalRisk)

    // Example: print first few values for verification
    println("x: ${xValues.take(5)}")
    println("Expected Risk: ${expectedRiskValues.take(5)}")
    println("Empirical Risk: ${empiricalRiskValues.take(5)}")
}

/**
 * Plots gradient descent trajectory over a function.
 *
 * @param fLine The range of x values for the function line.
 * @param res The x values at each step of the gradient descent.
 * @param func The function to plot.
 * @param width The plot width in pixels.
 * @param height The plot height in pixels.
 * @return A Plot object with the function line and gradient descent trajectory.
 */
fun plotGD(
    fLine: List<Float>,
    res: List<Float>,
    func: (Float) -> Float,
    width: Int,
    height: Int,
): Plot {
    var data =
        mapOf(
            "x" to fLine,
            "fx" to fLine.map(func),
        )
    var data1 =
        mapOf(
            "f" to res.map(func),
            "x1" to res,
        )
    val plot =
        letsPlot() +
            geomLine(data = data) {
                x = "x"
                y = "fx"
            } +
            geomLine(data = data1, color = "red") {
                x = "x1"
                y = "f"
            } +
            geomPoint(data = data1, size = 3.0) {
                x = "x1"
                y = "f"
            }
    return plot + ggsize(width, height)
}

/**
 * Container class for optimization-related utilities and demonstrations.
 */
class Optimization
