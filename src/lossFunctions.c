#include "libs.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <time.h>

double categoryCrossEntropy(double target[], double output[], int size){
    double result = 0.0f;
    for (int i=0; i<size; i++) printf("%f,", log(output[i]));
    printf("\n");
    for (int i=0; i<size; i++) printf("%f,", target[i]);
    printf("\n");
    for (int i=0; i<size;i++){
        result += (target[i] * log(output[i]));
    }
    return 0 - result;
}