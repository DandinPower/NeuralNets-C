#include "libs.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <time.h>

double categoryCrossEntropy(double target[], double output[], int size){
    double result = 0.0f;
    for (int i=0; i<size;i++){
        double tempOutput = output[i];
        if (fabs(tempOutput) < 1e-6) tempOutput = 1e-10;
        result += (target[i] * log(tempOutput));
    }
    return 0 - result;
}

void dCrossAndSoftmax(double delta[], double target[], double output[], int size){
    for (int i=0; i<size; i++) delta[i] = output[i] - target[i];
}