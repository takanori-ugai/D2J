package jp.live.ugai.d2j.util

import ai.djl.ndarray.NDManager
import org.jetbrains.letsPlot.geom.geomContour
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.geom.geomPoint
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.letsPlot

/**
 * Singleton for GradDescUtils.
 */
object GradDescUtils {
    /**
     * Executes plotGD.
     */
    fun plotGD(
        x0: List<Float>,
        y0: List<Float>,
        segment: List<Float>,
        func: (Float) -> Float,
        width: Int,
        height: Int,
    ): Plot {
        // Function Line
        val data = mapOf("x" to x0, "y" to y0, "y1" to segment.map(func))
        var plot = letsPlot(data = data)
        plot +=
            geomLine {
                x = "x"
                y = "y"
            }
        plot +=
            geomLine {
                x = "x"
                y = "y1"
            }
        plot +=
            geomPoint {
                x = "x"
                y = "y1"
            }
        return plot + ggsize(width, height)
    }

    /**
     * Executes showTrace.
     */
    fun showTrace(
        res: List<Float>,
        f: (Float) -> Float,
        manager: NDManager,
    ): Plot {
        var n = res.map { Math.abs(it) }.max()
        val fLine = manager.arange(-n, n, 0.01f).toFloatArray().toList()
        return plotGD(fLine, fLine.map(f), res, f, 500, 400)
    }

    // Optimize a 2D objective function with a customized trainer.

    /**
     * Executes train2d.
     */
    fun train2d(
        trainer: (List<Float>) -> List<Float>,
        steps: Int,
    ): List<Weights> {
        // s1 and s2 are internal state variables and will
        // be used later in the chapter
        var x1 = -5f
        var x2 = -2f
        var s1 = 0f
        var s2 = 0f
        val results = mutableListOf<Weights>()
        results.add(Weights(x1, x2))
        for (i in 1 until steps + 1) {
            val step: List<Float> = trainer(listOf(x1, x2, s1, s2))
            x1 = step[0]
            x2 = step[1]
            s1 = step[2]
            s2 = step[3]
            results.add(Weights(x1, x2))
        }
        return results
    }

    // Show the trace of 2D variables during optimization.

    /**
     * Executes showTrace2d.
     */
    fun showTrace2d(
        f: (Float, Float) -> Float,
        results: List<Weights>,
    ): Plot {
        // Contour and meshgrid rendering depends on tablesaw support.
        println("Tablesaw not supporting for contour and meshgrids, will update soon")

        fun meshgridPoints(
            x: List<Double>,
            y: List<Double>,
        ): List<Pair<Double, Double>> {
            val xSer = y.flatMap { x }
            var yInd = -1
            val ySer =
                y.flatMap {
                    yInd++
                    List(x.size) { y[yInd] }
                }
            return xSer.zip(ySer)
        }

        val x1 = results.map { it.x1 }
        val x2 = results.map { it.x2 }

        var gridPoints =
            meshgridPoints(
                generateSequence(
                    (x1.min() - 0.5f).toDouble(),
                    { it + 15.0 / 50 },
                ).takeWhile { it <= (x1.max() + 0.5f).toDouble() }.toList(),
                generateSequence(
                    (x2.min() - 0.5f).toDouble(),
                    { it + 15.0 / 40 },
                ).takeWhile { it <= (x2.max() + 0.5f).toDouble() }.toList(),
            )

        var dd = mapOf("x" to x1, "y" to x2, "z" to results.map { f(it.x1, it.x2) })
        var dd3 =
            mapOf(
                "x" to gridPoints.map { it.first },
                "y" to gridPoints.map { it.second },
                "z" to gridPoints.map { f(it.first.toFloat(), it.second.toFloat()) },
            )

        var plot = letsPlot()
        plot +=
            geomLine(data = dd) {
                x = "x"
                y = "y"
            }
        plot +=
            geomPoint(data = dd, size = 3) {
                x = "x"
                y = "y"
            }
        plot +=
            geomContour(data = dd3, color = "red", binWidth = 7.0) {
                x = "x"
                y = "y"
                z = "z"
            }
        return plot + ggsize(500, 400)
    }

    /**
     * Executes plotGammas.
     */
    fun plotGammas(
        time: List<Float>,
        gammas: List<Float>,
        width: Int,
        height: Int,
    ): Plot {
        val gamma1 = mutableListOf<Double>()
        val gamma2 = mutableListOf<Double>()
        val gamma3 = mutableListOf<Double>()
        val gamma4 = mutableListOf<Double>()

        // Calculate all gammas over time
        for (i in time.indices) {
            gamma1.add(Math.pow(gammas[0].toDouble(), i.toDouble()))
            gamma2.add(Math.pow(gammas[1].toDouble(), i.toDouble()))
            gamma3.add(Math.pow(gammas[2].toDouble(), i.toDouble()))
            gamma4.add(Math.pow(gammas[3].toDouble(), i.toDouble()))
        }

        var data = mapOf("x" to time, "y1" to gamma1, "y2" to gamma2, "y3" to gamma3, "y4" to gamma4)
        var plot = letsPlot(data = data)
        plot +=
            geomLine {
                x = "x"
                y = "y1"
            }
        plot +=
            geomLine {
                x = "x"
                y = "y2"
            }
        plot +=
            geomLine {
                x = "x"
                y = "y3"
            }
        plot +=
            geomLine {
                x = "x"
                y = "y4"
            }
        return plot + ggsize(width, height)
    }
}

/**
 * Represents Weights.
 * @property x1 The x1.
 * @property x2 The x2.
 */
class Weights(
    /**
     * The x1.
     */
    var x1: Float,
    /**
     * The x2.
     */
    var x2: Float,
)
