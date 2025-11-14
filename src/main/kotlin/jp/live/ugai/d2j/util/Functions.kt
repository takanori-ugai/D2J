package jp.live.ugai.d2j.util

import ai.djl.Device
import ai.djl.engine.Engine

/**
 * Returns the i'th GPU [Device] if available, otherwise returns the CPU [Device].
 */
fun tryGpu(i: Int): Device = if (Engine.getInstance().gpuCount > i) Device.gpu(i) else Device.cpu()
