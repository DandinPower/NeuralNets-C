#include "libs.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <time.h>

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
    double sign;
    if ((int)rand() % 2 == 1) sign = -1.0f;
    else sign = 1.0f;
    return sign * ((double)rand())/((double)RAND_MAX); 
}

//初始化訓練order
void InitOrder(int arr[], int size){
    for (int i=0; i < size; i++){
        arr[i] = i;
    }
}