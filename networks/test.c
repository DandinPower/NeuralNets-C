#include "../src/activations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

int main(void){
    double x = 0.1f;
    double y = relu(x);
    printf("%f\n",y);
}