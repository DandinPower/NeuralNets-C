#include "activations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Activation function and its derivative
double sigmoid(double x) { 
    return 1 / (1 + exp(-x)); 
}
double dSigmoid(double x) { 
    return x * (1 - x); 
}
double relu(double x) {
    if (x >= 0) return x;
    else return 0;
}
double dRelu(double x) {
    if (x > 0) return 1;
    else return 0;
}
//回傳totalexponent
double softmax(double arr[], int size){
    double totalExponential = 0;
    double exponential[size];
    int exception = 0;
    int maxIndex = 0;
    double max = arr[0];
    for (int i=0; i < size; i++){
        if (arr[i] >= max) {
            max = arr[i];
            maxIndex = i;
        }
        double expResult = exp(arr[i]);
        exponential[i] = expResult;
        totalExponential += expResult;
    }
    if (isinf(totalExponential)) exception = 1;
    if (exception){
        for (int i=0; i < size; i++)arr[i] = 0.0f;
        arr[maxIndex] = 1.0f;
        printf("boom!\n");
    }
    else{
        for (int i=0; i < size; i++){
            double temp = exponential[i] / totalExponential;
            arr[i] = temp;
        }
    }
    return totalExponential;        
}

void dSoftmax(double delta[], double output[], int size, double totalExponential){
    for (int i=0; i<size; i++){
        delta[i] = exp(output[i]) / ((totalExponential - output[i]) + exp(output[i]) * exp(output[i]));
    }
}