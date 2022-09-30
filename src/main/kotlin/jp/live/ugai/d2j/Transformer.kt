package jp.live.ugai.d2j

import ai.djl.ndarray.NDManager
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.ParameterStore

fun main() {
    val manager = NDManager.newBaseManager()
    val ps = ParameterStore(manager, false)
    fun positionWiseFFN(ffn_num_hiddens: Long, ffn_num_outputs: Long) : AbstractBlock {
        val net = SequentialBlock()
        net.add(Linear.builder().setUnits(ffn_num_hiddens).build())
        net.add(Activation::relu)
        net.add(Linear.builder().setUnits(ffn_num_outputs).build());
        return net
    }
//    net.setInitializer(NormalInitializer(), Parameter.Type.WEIGHT)
}

class Transformer {
}