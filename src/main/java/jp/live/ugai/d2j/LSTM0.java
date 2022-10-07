package jp.live.ugai.d2j;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.recurrent.RecurrentBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.Preconditions;

public class LSTM0 extends RecurrentBlock {

  /**
   * Creates an LSTM block.
   *
   * @param builder the builder used to create the RNN block
   */
  LSTM0(LSTM0.Builder builder) {
    super(builder);
    gates = 4;
  }

  /** {@inheritDoc} */
  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> params) {
    NDArrayEx ex = inputs.head().getNDArrayInternal();
    Device device = inputs.head().getDevice();
    NDList rnnParams = new NDList();
    for (Parameter parameter : parameters.values()) {
      rnnParams.add(parameterStore.getValue(parameter, device, training));
    }

    NDArray input = inputs.head();
    if (inputs.size() == 1) {
      int batchIndex = batchFirst ? 0 : 1;
      Shape stateShape =
          new Shape((long) numLayers * getNumDirections(), input.size(batchIndex), stateSize);
      // hidden state
      inputs.add(input.getManager().zeros(stateShape));
      // cell
      inputs.add(input.getManager().zeros(stateShape));
    }
    if (inputs.size() == 2) {
      int batchIndex = batchFirst ? 0 : 1;
      Shape stateShape =
          new Shape((long) numLayers * getNumDirections(), input.size(batchIndex), stateSize);
      // hidden state
      //            inputs.add(input.getManager().zeros(stateShape));
      // cell
      inputs.add(input.getManager().zeros(stateShape));
    }
    NDList outputs =
        ex.lstm(
            input,
            new NDList(inputs.get(1), inputs.get(2)),
            rnnParams,
            hasBiases,
            numLayers,
            dropRate,
            training,
            bidirectional,
            batchFirst);
    if (returnState) {
      return outputs;
    }
    outputs.stream().skip(1).forEach(NDArray::close);
    return new NDList(outputs.get(0));
  }

  /**
   * Creates a builder to build a {@link ai.djl.nn.recurrent.LSTM}.
   *
   * @return a new builder
   */
  public static LSTM0.Builder builder() {
    return new LSTM0.Builder();
  }

  /** The Builder to construct a {@link ai.djl.nn.recurrent.LSTM} type of {@link Block}. */
  public static final class Builder extends BaseBuilder<LSTM0.Builder> {

    /** {@inheritDoc} */
    @Override
    protected LSTM0.Builder self() {
      return this;
    }

    /**
     * Builds a {@link ai.djl.nn.recurrent.LSTM} block.
     *
     * @return the {@link ai.djl.nn.recurrent.LSTM} block
     */
    public LSTM0 build() {
      Preconditions.checkArgument(
          stateSize > 0 && numLayers > 0, "Must set stateSize and numStackedLayers");
      return new LSTM0(this);
    }
  }
}
