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