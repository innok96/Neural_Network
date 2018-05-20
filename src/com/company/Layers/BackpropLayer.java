package com.company.Layers;

public interface BackpropLayer extends Layer {
    void randomize(double min,double max);

    double[] computeBackwardError(double[] input,double[] error);

    void adjust(double[] input,double[] error,double rate,double momentum);
}