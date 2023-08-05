package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.geom.geomPoint
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.letsPlot

fun main() {
    val f = { x: Float -> x * Math.cos(Math.PI * x).toFloat() }
    val g = { x: Float -> f(x) + 0.2f * Math.cos(5 * Math.PI * x).toFloat() }

    val manager = NDManager.newBaseManager()
    val X = manager.arange(0.5f, 1.5f, 0.01f)
    val x = X.toFloatArray().toList()
    val fx = x.map(f)
    val gx = x.map(g)

    val group1 = List(x.size) { "Expected Risk" }
    val group2 = List(x.size) { "Empirical Risk" }
}

fun plotGD(fLine: List<Float>, res: List<Float>, func: (Float) -> Float, width: Int, height: Int): Plot {
    var data = mapOf(
        "x" to fLine,
        "fx" to fLine.map(func)
    )
    var data1 = mapOf(
        "f" to res.map(func),
        "x1" to res
    )
    val plot = letsPlot() +
        geomLine(data = data) { x = "x"; y = "fx" } +
        geomLine(data = data1, color = "red") { x = "x1"; y = "f" } +
        geomPoint(data = data1, size = 3.0) { x = "x1"; y = "f" }
    return plot + ggsize(width, height)
}

class Optimization
