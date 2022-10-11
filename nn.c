#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4
#define lr 0.1
#define epochs 10000

double hiddenLayer[numHiddenNodes];
double outputLayer[numOutputs];
double hiddenLayerBias[numHiddenNodes];
double outputLayerBias[numOutputs];
double hiddenWeights[numInputs][numHiddenNodes];
double outputWeights[numHiddenNodes][numOutputs];

void shuffle(int arr[], int size){
    srand(time(0));
    for (int i = 0; i < size; i++) {
        int j = rand() % size;
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }
}

// Activation function and its derivative
double sigmoid(double x) { 
    return 1 / (1 + exp(-x)); 
}
double dSigmoid(double x) { 
    return x * (1 - x); 
}

// Init all weights and biases between 0.0 and 1.0
double init_weight() { 
    return ((double)rand())/((double)RAND_MAX); 
}

void init_hidden_layer(){
    for (int i=0; i < numHiddenNodes; i++){
        hiddenLayerBias[i] = init_weight();
    }
    for (int i=0; i< numInputs; i++){
        for (int j=0; j<numHiddenNodes; j++){
            hiddenWeights[i][j] = init_weight();
        }
    }
}

void init_output_layer(){
    for (int i=0; i < numOutputs; i++){
        outputLayerBias[i] = init_weight();
    }
    for (int i=0; i< numHiddenNodes; i++){
        for (int j=0; j<numOutputs; j++){
            outputWeights[i][j] = init_weight();
        }
    }
}

double training_inputs[numTrainingSets][numInputs] = { {0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f} };
double training_outputs[numTrainingSets][numOutputs] = { {0.0f},{1.0f},{1.0f},{0.0f} };

void inference(){
    for (int i=0; i< numTrainingSets; i++){
        for (int j=0; j<numHiddenNodes; j++) {
            double activation=hiddenLayerBias[j];
            for (int k=0; k<numInputs; k++) {
                activation+=training_inputs[i][k]*hiddenWeights[i][j];
            }
            hiddenLayer[j] = sigmoid(activation);
        }
        // Compute output layer activation
        for (int j=0; j<numOutputs; j++) {
            double activation=outputLayerBias[j];
            for (int k=0; k<numHiddenNodes; k++) {
                activation+=hiddenLayer[k]*outputWeights[k][j];
            }
            outputLayer[j] = sigmoid(activation);
        }
        printf("[%d]inference output: %f\n",i,outputLayer[0]);
    }
}

int main(){
    init_output_layer();
    init_hidden_layer();
    // Iterate through the entire training for a number of epochs
    for (int n=0; n < epochs; n++) {
        // As per SGD, shuffle the order of the training set
        int trainingSetOrder[] = {0,1,2,3};
        shuffle(trainingSetOrder,numTrainingSets);
        // Cycle through each of the training set elements
        for (int x=0; x<numTrainingSets; x++) {
            int i = trainingSetOrder[x];
            // Compute hidden layer activation
            for (int j=0; j<numHiddenNodes; j++) {
                double activation=hiddenLayerBias[j];
                for (int k=0; k<numInputs; k++) {
                    activation+=training_inputs[i][k]*hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            // Compute output layer activation
            for (int j=0; j<numOutputs; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<numHiddenNodes; k++) {
                    activation+=hiddenLayer[k]*outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            // Compute change in output weights
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                double dError = (training_outputs[i][j]-outputLayer[j]);
                deltaOutput[j] = dError*dSigmoid(outputLayer[j]);
            }
            // Compute change in hidden weights
            double deltaHidden[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                double dError = 0.0f;
                for(int k=0; k<numOutputs; k++) {
                    dError+=deltaOutput[k]*outputWeights[j][k];
                }
                deltaHidden[j] = dError*dSigmoid(hiddenLayer[j]);
            }
            // Apply change in output weights
            for (int j=0; j<numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j]*lr;
                for (int k=0; k<numHiddenNodes; k++) {
                    outputWeights[k][j]+=hiddenLayer[k]*deltaOutput[j]*lr;
                }
            } 
        }
    }
    inference();
    return 0;
}