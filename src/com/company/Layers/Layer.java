package com.company.Layers;

import java.io.Serializable;

public interface Layer extends Serializable {
    int getInputSize();

    int getSize();

    double[] computeOutput(double[] input);
}