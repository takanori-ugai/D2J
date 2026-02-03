package jp.live.ugai.d2j

fun flattenParametersIfAvailable(block: Any) {
    val method =
        block.javaClass.methods.firstOrNull { candidate ->
            candidate.name == "flattenParameters" && candidate.parameterCount == 0
        }
    method?.invoke(block)
}
