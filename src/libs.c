#include "libs.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

//陣列打亂
void shuffle(int arr[], int size){
    srand(time(0));
    for (int i = 0; i < size; i++) {
        int j = rand() % size;
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }
}

//隨機產生 -1~1的weight
double GetRandomWeight() { 
    return ((double)rand())/((double)RAND_MAX); 
}