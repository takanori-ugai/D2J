package jp.live.ugai.d2j

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape

/**
 * Mimics MXNet-style sequence masking for DJL PyTorch NDArrays.
 *
 * @param input Input tensor to mask.
 * @param validLen 1D tensor of sequence lengths for each batch element.
 * @param value Optional fill value for masked positions (defaults to zero).
 * @return Array with tokens beyond `validLen` masked out.
 */
fun sequenceMask(
    input: NDArray,
    validLen: NDArray,
    value: Float = 0f,
): NDArray {
    require(input.shape.dimension() >= 2) { "sequenceMask expects input with at least 2 dimensions" }

    val manager = input.manager
    val batch = input.shape[0]
    val steps = input.shape[1]

    val stepsRange =
        manager.arange(steps.toInt()).reshape(Shape(1, steps)).broadcast(Shape(batch, steps))
    val lenBroadcast =
        validLen
            .toType(DataType.INT64, false)
            .reshape(batch, 1)
            .broadcast(Shape(batch, steps))
    val mask2d = stepsRange.lt(lenBroadcast)
    val mask =
        if (input.shape.dimension() > 2) {
            mask2d.expandDims(-1).broadcast(input.shape)
        } else {
            mask2d
        }

    val maskTyped = mask.toType(input.dataType, false)
    val kept = input.mul(maskTyped)
    if (value == 0f) {
        return kept
    }
    val fill = maskTyped.neg().add(1).mul(value)
    return kept.add(fill)
}
