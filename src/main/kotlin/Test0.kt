fun main() {
    val v1: Int = 0

    class T(
        val v1: Int,
    ) {
        fun f0(): Int = v1
    }

    val t0 = T(1)
    println(t0.f0())
}

class Test0
