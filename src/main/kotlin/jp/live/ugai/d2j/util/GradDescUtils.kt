package jp.live.ugai.d2j.util

import ai.djl.ndarray.NDManager
import org.jetbrains.letsPlot.geom.geomContour
import org.jetbrains.letsPlot.geom.geomLine
import org.jetbrains.letsPlot.geom.geomPoint
import org.jetbrains.letsPlot.ggsize
import org.jetbrains.letsPlot.intern.Plot
import org.jetbrains.letsPlot.letsPlot

/**
 * Utilities for plotting and tracing gradient descent behavior.
 */
object GradDescUtils {
    /**
     * Plots a function line and gradient descent trajectory.
     *
     * @param x0 The x-values for the function line.
     * @param y0 The y-values for the function line.
     * @param segment The x-values visited during optimization.
     * @param func The objective function to plot along the segment.
     * @param width The plot width in pixels.
     * @param height The plot height in pixels.
     * @return A Plot with the function line and optimization trajectory.
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
     * Builds a plot showing the optimization trace over a 1D function.
     *
     * @param res The x-values visited during optimization.
     * @param f The objective function to evaluate.
     * @param manager The NDManager used to create the plotting range.
     * @return A Plot of the function and trace points.
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
     * Runs a 2D optimization loop with a custom trainer function.
     *
     * @param trainer The update rule that returns the next state from current state.
     * @param steps The number of optimization steps to run.
     * @return The list of 2D points visited during optimization.
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
     * Plots the trace of 2D optimization steps on a contour plot.
     *
     * @param f The 2D objective function.
     * @param results The optimization trace points.
     * @return A Plot showing the contour and optimization trajectory.
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
     * Plots exponential decay curves for multiple gamma values over time.
     *
     * @param time The time steps.
     * @param gammas The gamma values to plot.
     * @param width The plot width in pixels.
     * @param height The plot height in pixels.
     * @return A Plot of gamma decay curves.
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
 * Represents a point in 2D optimization space.
 *
 * @property x1 The first coordinate.
 * @property x2 The second coordinate.
 */
data class Weights(
    val x1: Float,
    val x2: Float,
)
