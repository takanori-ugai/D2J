package jp.live.ugai.d2j.util

import ai.djl.modality.cv.Image
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.output.Rectangle
import ai.djl.ndarray.NDArray
import java.awt.Color
import java.awt.RenderingHints
import java.awt.image.BufferedImage

object ImageUtils {
    fun showImages(images: List<BufferedImage>, labels: List<String>, width: Int, height: Int): BufferedImage {
        val col = Math.min(1280 / width, images.size)
        val row = (images.size + col - 1) / col
        val textHeight = 28
        val w = col * (width + 3)
        val h = row * (height + 3) + textHeight
        val output = BufferedImage(w + 3, h + 3, BufferedImage.TYPE_INT_RGB)
        val g = output.createGraphics()
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.paint = Color.LIGHT_GRAY
        g.fill(java.awt.Rectangle(0, 0, w + 3, h + 3))
        g.paint = Color.BLACK
        val font = g.font
        val metrics = g.getFontMetrics(font)
        for (i in images.indices) {
            val x = i % col * (width + 3) + 3
            val y = i / col * (height + 3) + 3
            val tx = x + (width - metrics.stringWidth(labels[i])) / 2
            val ty = y + (textHeight - metrics.height) / 2 + metrics.ascent
            g.drawString(labels[i], tx, ty)
            val img = images[i]
            g.drawImage(img, x, y + textHeight, width, height, null)
        }
        g.dispose()
        return output
    }

    fun showImages(images: List<BufferedImage>, width: Int, height: Int): BufferedImage {
        val col = Math.min(1280 / width, images.size)
        val row = (images.size + col - 1) / col
        val w = col * (width + 3)
        val h = row * (height + 3)
        val output = BufferedImage(w + 3, h + 3, BufferedImage.TYPE_INT_RGB)
        val g = output.createGraphics()
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.paint = Color.LIGHT_GRAY
        g.fill(java.awt.Rectangle(0, 0, w + 3, h + 3))
        for (i in images.indices) {
            val x = i % col * (width + 3) + 3
            val y = i / col * (height + 3) + 3
            val img = images[i]
            g.drawImage(img, x, y, width, height, null)
        }
        g.dispose()
        return output
    }

    fun drawBBoxes(img: Image, boxes: NDArray, labels: List<String>?) {
        var labels = labels
        if (labels == null) {
            labels = List(boxes.size(0).toInt()) { "" }
        }
        val classNames: MutableList<String> = mutableListOf()
        val prob: MutableList<Double> = mutableListOf()
        val boundBoxes: MutableList<Rectangle> = mutableListOf()
        for (i in 0 until boxes.size(0)) {
            val box = boxes[i]
            val rect: Rectangle = bboxToRect(box)
            classNames.add(labels[i.toInt()])
            prob.add(1.0)
            boundBoxes.add(rect)
        }
        val detectedObjects = DetectedObjects(classNames, prob, boundBoxes.toList())
        img.drawBoundingBoxes(detectedObjects)
    }

    fun bboxToRect(bbox: NDArray): Rectangle {
        val width = bbox.getFloat(2) - bbox.getFloat(0)
        val height = bbox.getFloat(3) - bbox.getFloat(1)

        return Rectangle(
            bbox.getFloat(0).toDouble(),
            bbox.getFloat(1).toDouble(),
            width.toDouble(),
            height.toDouble()
        )
    }
}
