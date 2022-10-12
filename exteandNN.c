#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define RELU 0
#define SIGMOID 1
#define SOFTMAX 2

typedef struct {
    int numInput;
    double* weight;
    double bias;
    double* inputs;
    double* nextDelta;
    double* delta;
}node_t;

typedef struct {
    int numInput;
    int numOutput;
    double* delta;
    node_t* nodes;
    int activation;
}linearlayer_t;

double GetRandomWeight() { 
    return ((double)rand())/((double)RAND_MAX); 
}

void InitWeight(node_t *x){
    x->weight = (double*)malloc(sizeof(double)* x->numInput);
}

void ResetWeight(node_t *x){
    for (int i=0; i<x->numInput; i++) x->weight[i] = GetRandomWeight();
    x->bias = GetRandomWeight();
}

void InitInputs(node_t *x){
    x->inputs = (double*)malloc(sizeof(double)* x->numInput);
}

void ResetInputs(node_t *x){
    for (int i=0; i<x->numInput; i++) x->inputs[i] = 0; 
}

void InitDelta(node_t *x){
    x->delta = (double*)malloc(sizeof(double)* x->numInput);
}

void ResetDelta(node_t *x){
    for (int i=0; i<x->numInput; i++) x->delta[i] = 0; 
}

void ShowWeight(node_t x){
    for (int i=0; i<x.numInput; i++) printf("%f, ",x.weight[i]);
    printf("\n");
}

node_t GetNewNode(int numInput){
    node_t newNode;
    newNode.numInput = numInput;
    InitInputs(&newNode);
    ResetInputs(&newNode);
    InitDelta(&newNode);
    ResetDelta(&newNode);
    InitWeight(&newNode);
    ResetWeight(&newNode);
    return newNode;
}

void InitLayerDelta(linearlayer_t *x){
    x->delta = (double*)malloc(sizeof(double)* x->numInput);
}

void ResetLayerDelta(linearlayer_t *x){
    for (int i=0; i<x->numInput; i++) x->delta[i] = 0; 
}


int main(){
    /*
    linearlayer_t a;
    a.numInput = 3;
    a.numOutput = 5;
    a.activation = RELU;
    */
    node_t a;
    a.numInput = 10;
    InitDelta(&a);
    ResetDelta(&a);

    return 0;
}