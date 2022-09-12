package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager

fun main() {
    val f = { x: Float -> x * Math.cos(Math.PI * x).toFloat() }
    val g = { x: Float -> f(x) + 0.2f * Math.cos(5 * Math.PI * x).toFloat() }

    val manager = NDManager.newBaseManager()
    val X = manager.arange(0.5f, 1.5f, 0.01f)
    val x = X.toFloatArray().toList()
    val fx = x.map(f)
    val gx = x.map(g)

    val group1 = Array<String>(x.size) { "Expected Risk" }.toList()
    val group2 = Array<String>(x.size) { "Empirical Risk" }.toList()
}

class Optimization
