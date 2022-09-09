package jp.live.ugai.d2j.util

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.nn.AbstractBlock

object TrainingChapter9 {
    /** Clip the gradient.  */
    fun gradClipping(net: Any, theta: Int, manager: NDManager) {
        var result = 0.0
        val params: NDList
        params = NDList()
        for (pair in (net as AbstractBlock).parameters) {
            params.add(pair.value.array)
        }
        for (p in params) {
            val gradient = p.gradient.stopGradient()
            gradient.attach(manager)
            result += gradient.pow(2.0).sum().getFloat().toDouble()
        }
        val norm = Math.sqrt(result)
        if (norm > theta) {
            for (param in params) {
                val gradient = param.gradient
                gradient.muli(theta / norm)
            }
        }
    }
}
