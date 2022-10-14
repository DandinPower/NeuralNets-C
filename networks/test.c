#include "../src/activations.h"
#include "../src/libs.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define DENSE_1_INPUT 768
#define DENSE_1_OUTPUT 256
#define DENSE_2_INPUT 256
#define DENSE_2_OUTPUT 64
#define DENSE_3_INPUT 64
#define DENSE_3_OUTPUT 10
#define numTrainingSets 60000
double training_inputs[numTrainingSets][DENSE_1_INPUT];
double training_outputs[numTrainingSets][DENSE_3_OUTPUT];

#define N 5000
int main() {
    printf("%f\n",GetRandomWeight());

    return 0;
}