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
    for (int i=0; i < size; i++){
        double expResult = exp(arr[i]);
        exponential[i] = expResult;
        totalExponential += expResult;
    }
    for (int i=0; i < size; i++){
        arr[i] = exponential[i] / totalExponential;
    }
    return totalExponential;        
}