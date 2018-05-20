package com.company;

import com.company.Layers.BackpropLayer;
import com.company.Layers.Layer;

public final class BackpropNetwork extends Network {
    public BackpropNetwork(Layer[] layers) {
        super(layers);
        randomize(0,0.00003);
    }

    public void randomize(double min,double max) {
        final int size = getSize();
        for (int i = 0; i < size; i++) {
            Layer layer = getLayer(i);
            if (layer instanceof BackpropLayer) ((BackpropLayer)layer).randomize(min,max);
        }
    }

    public double learnPattern(double[] input,double[] goal,double rate,double momentum) {
        if (input == null || input.length != getInputSize() ||
                goal == null || goal.length != getOutputSize()) throw new IllegalArgumentException();

        final int size = getSize();
        double[][] outputs = new double[size][];

        outputs[0] = getLayer(0).computeOutput(input);
        for (int i = 1; i < size; i++)
            outputs[i] = getLayer(i).computeOutput(outputs[i - 1]);

        Layer layer = getLayer(size - 1);
        final int layerSize = layer.getSize();
        double[] error = new double[layerSize];
        double totalError = 0;

        for (int i = 0; i < layerSize; i++) {
            error[i] = (goal[i] - outputs[size - 1][i])*(1 - outputs[size - 1][i])*outputs[size-1][i];
            totalError += Math.abs(goal[i] - outputs[size - 1][i]);
        }

        if (layer instanceof BackpropLayer)
            ((BackpropLayer)layer).adjust(size == 1 ? input : outputs[size - 2],error,rate,momentum);

        double[] prevError = error;
        Layer prevLayer = layer;

        for (int i = size - 2; i >= 0; i--,prevError = error,prevLayer = layer) {
            layer = getLayer(i);
            if (prevLayer instanceof BackpropLayer)
                error = ((BackpropLayer)prevLayer).computeBackwardError(outputs[i],prevError);
            else
                error = prevError;
            if (layer instanceof BackpropLayer)
                ((BackpropLayer)layer).adjust(i == 0 ? input : outputs[i - 1],error,rate,momentum);
        }

        return totalError;
    }
}