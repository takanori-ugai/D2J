package jp.live.ugai.d2j.attention

import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import jp.live.ugai.d2j.lstm.Decoder

abstract class AttentionDecoder : Decoder() {
    var attentionWeightArr: MutableList<Pair<FloatArray, Shape>> = mutableListOf()
    abstract override fun initState(encOutputs: NDList): NDList
    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        throw UnsupportedOperationException("Not implemented")
    }
}
