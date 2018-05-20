package com.company.Layers;

import com.company.Layers.BackpropLayer;

public final class SigmoidLayer implements BackpropLayer {
    private final int WEIGHT = 0;

    private final int DELTA = 1;

    public SigmoidLayer(int inputSize, int size) {
        if (inputSize < 1 || size < 1) throw new IllegalArgumentException();

        matrix = new double[size][inputSize + 1][2];

        this.inputSize = inputSize;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getSize() {
        return matrix.length;
    }

    public double[] computeOutput(double[] input) {
        if (input == null || input.length != inputSize)
            throw new IllegalArgumentException();

        final int size = matrix.length;
        double[] output = new double[size];
        for (int i = 0; i < size; i++) {
            output[i] = matrix[i][0][WEIGHT];
            for (int j = 0; j < inputSize; j++)
                output[i] += input[j] * matrix[i][j + 1][WEIGHT];
            output[i] = 1 / (1 + Math.exp(-output[i]));
        }

        return output;
    }

    public void randomize(double min, double max) {
        final int size = matrix.length;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < inputSize + 1; j++) {
                matrix[i][j][WEIGHT] = min + (max - min) * Math.random();
                matrix[i][j][DELTA] = 0;
            }
        }
    }

    public double[] computeBackwardError(double[] input, double[] error) {
        if (input == null || input.length != inputSize ||
                error == null || error.length != matrix.length) throw new IllegalArgumentException();

        double[] output = computeOutput(input);
        final int size = matrix.length;
        double[] backwardError = new double[inputSize];

        for (int i = 0; i < inputSize; i++) {
            backwardError[i] = 0;
            for (int j = 0; j < size; j++)
                backwardError[i] += error[j] * matrix[j][i + 1][WEIGHT];
            backwardError[i] *= (input[i] * (1 - input[i]));
        }

        return backwardError;
    }

    public void adjust(double[] input, double[] error, double rate, double momentum) {
        if (input == null || input.length != inputSize ||
                error == null || error.length != matrix.length) throw new IllegalArgumentException();

        double[] output = computeOutput(input);
        final int size = matrix.length;

        for (int i = 0; i < size; i++) {
            final double grad = error[i] * (output[i] * (1 - output[i]));
            matrix[i][0][DELTA] = rate * grad + momentum * matrix[i][0][DELTA];
            matrix[i][0][WEIGHT] += matrix[i][0][DELTA];
            for (int j = 0; j < inputSize; j++) {
                matrix[i][j + 1][DELTA] = (1 - momentum) * rate * input[j] * grad + momentum * matrix[i][j + 1][DELTA];
                matrix[i][j + 1][WEIGHT] += matrix[i][j + 1][DELTA];
            }
        }
    }

    private final int inputSize;

    private double[][][] matrix;
}