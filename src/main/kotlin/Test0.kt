fun main() {
    val v1: Int = 0

    class t(val v1: Int) {
        fun f0(): Int {
            return v1
        }
    }

    val t0 = t(1)
    println(t0.f0())
}

class Test0
