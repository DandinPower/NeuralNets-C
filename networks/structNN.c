#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define LINEAR 0
#define RELU 1
#define SIGMOID 2
#define SOFTMAX 3

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

void InitLayerNode(linearlayer_t *x){
    x->nodes = (node_t*)malloc(sizeof(node_t)* x->numInput);
}

void ResetLayerNode(linearlayer_t *x){
    for (int i=0; i<x->numInput; i++) {
        x->nodes[i] = GetNewNode(x->numInput);
    }
}

linearlayer_t GetNewLayer(int numInput, int numOutput, int activation){
    linearlayer_t newLayer;
    newLayer.numInput = numInput;
    newLayer.numOutput = numOutput;
    newLayer.activation = activation;
    InitLayerDelta(&newLayer);
    ResetLayerDelta(&newLayer);
    ResetLayerNode(&newLayer);
    return newLayer;
}

int main(){
    linearlayer_t dense_1 = GetNewLayer(2, 3, RELU);
    return 0;
}