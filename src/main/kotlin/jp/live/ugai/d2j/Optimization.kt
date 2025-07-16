package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.geom.geomPoint
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.letsPlot

fun main() {
    // Define the target functions
    val expectedRisk: (Float) -> Float = { x -> x * Math.cos(Math.PI * x).toFloat() }
    val empiricalRisk: (Float) -> Float = { x -> expectedRisk(x) + 0.2f * Math.cos(5 * Math.PI * x).toFloat() }

    val manager = NDManager.newBaseManager()
    val xValues = manager.arange(0.5f, 1.5f, 0.01f).toFloatArray()
    val expectedRiskValues = xValues.map(expectedRisk)
    val empiricalRiskValues = xValues.map(empiricalRisk)

    // If plotting or further processing is needed, use these group labels
    val expectedGroup = List(xValues.size) { "Expected Risk" }
    val empiricalGroup = List(xValues.size) { "Empirical Risk" }

    // Example: print first few values for verification
    println("x: ${xValues.take(5)}")
    println("Expected Risk: ${expectedRiskValues.take(5)}")
    println("Empirical Risk: ${empiricalRiskValues.take(5)}")
}

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

class Optimization
