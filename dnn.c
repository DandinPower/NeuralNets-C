#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define LEARNING_RATE 0.1
#define EPOCHS 10000

#define DENSE_1_INPUT 2
#define DENSE_1_OUTPUT 4
#define DENSE_2_INPUT 4
#define DENSE_2_OUTPUT 2
#define DENSE_3_INPUT 2
#define DENSE_3_OUTPUT 1
#define numTrainingSets 4

double dense_1_Bias[DENSE_1_OUTPUT];
double dense_2_Bias[DENSE_2_OUTPUT];
double dense_3_Bias[DENSE_3_OUTPUT];
double dense_1_Weights[DENSE_1_INPUT][DENSE_1_OUTPUT];
double dense_2_Weights[DENSE_2_INPUT][DENSE_2_OUTPUT];
double dense_3_Weights[DENSE_3_INPUT][DENSE_3_OUTPUT];
double dense_1_output[DENSE_1_OUTPUT];
double dense_2_output[DENSE_2_OUTPUT];
double dense_3_output[DENSE_3_OUTPUT];
double delta_1[DENSE_1_INPUT];
double delta_2[DENSE_2_INPUT];
double delta_3[DENSE_3_INPUT];

double training_inputs[numTrainingSets][DENSE_1_INPUT] = { {0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f} };
double training_outputs[numTrainingSets][DENSE_3_OUTPUT] = { {0.0f},{1.0f},{1.0f},{0.0f} };

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

//隨機產生 -1~1的weight
double GetRandomWeight() { 
    return ((double)rand())/((double)RAND_MAX); 
}

//初始化DENSE_1參數
void InitDense1Layer(){
    for (int i=0; i < DENSE_1_OUTPUT; i++) dense_1_Bias[i] = GetRandomWeight();
    for (int i=0; i< DENSE_1_INPUT; i++) for (int j=0; j<DENSE_1_OUTPUT; j++) dense_1_Weights[i][j] = GetRandomWeight();
}

//初始化DENSE_2參數
void InitDense2Layer(){
    for (int i=0; i < DENSE_2_OUTPUT; i++) dense_2_Bias[i] = GetRandomWeight();
    for (int i=0; i< DENSE_2_INPUT; i++) for (int j=0; j<DENSE_1_OUTPUT; j++) dense_2_Weights[i][j] = GetRandomWeight();
}

//初始化DENSE_3參數
void InitDense3Layer(){
    for (int i=0; i < DENSE_3_OUTPUT; i++) dense_3_Bias[i] = GetRandomWeight();
    for (int i=0; i< DENSE_3_INPUT; i++) for (int j=0; j<DENSE_1_OUTPUT; j++) dense_3_Weights[i][j] = GetRandomWeight();
}

//inference模型
void inference(){
    int trainingSetOrder[] = {0,1,2,3};
    for (int x=0; x<numTrainingSets; x++) {
        int i = trainingSetOrder[x];
        // Compute dense1 activation
        for (int j=0; j<DENSE_1_OUTPUT; j++) {
            double activation=dense_1_Bias[j];
            for (int k=0; k<DENSE_1_INPUT; k++) {
                activation += training_inputs[i][k] * dense_1_Weights[k][j];
            }
            dense_1_output[j] = relu(activation);
        }
        // Compute dense2 activation
        for (int j=0; j<DENSE_2_OUTPUT; j++) {
            double activation=dense_2_Bias[j];
            for (int k=0; k<DENSE_2_INPUT; k++) {
                activation += training_inputs[i][k] * dense_2_Weights[k][j];
            }
            dense_2_output[j] = relu(activation);
        }
        // Compute dense3 activation
        for (int j=0; j<DENSE_3_OUTPUT; j++) {
            double activation=dense_3_Bias[j];
            for (int k=0; k<DENSE_3_INPUT; k++) {
                activation += training_inputs[i][k] * dense_3_Weights[k][j];
            }
            dense_3_output[j] = sigmoid(activation);
        }
        printf("[%d]inference output: %f\n",i,dense_3_output[0]);
    }
}

//訓練epoch
void Training(){
    int trainingSetOrder[] = {0,1,2,3};
    shuffle(trainingSetOrder,numTrainingSets);
    double loss = 0.0f;
    for (int x=0; x<numTrainingSets; x++) {
        int i = trainingSetOrder[x];
        // Compute dense1 activation
        for (int j=0; j<DENSE_1_OUTPUT; j++) {
            double activation=dense_1_Bias[j];
            for (int k=0; k<DENSE_1_INPUT; k++) {
                activation += training_inputs[i][k] * dense_1_Weights[k][j];
            }
            dense_1_output[j] = relu(activation);
        }
        // Compute dense2 activation
        for (int j=0; j<DENSE_2_OUTPUT; j++) {
            double activation=dense_2_Bias[j];
            for (int k=0; k<DENSE_2_INPUT; k++) {
                activation += training_inputs[i][k] * dense_2_Weights[k][j];
            }
            dense_2_output[j] = relu(activation);
        }
        // Compute dense3 activation
        for (int j=0; j<DENSE_3_OUTPUT; j++) {
            double activation=dense_3_Bias[j];
            for (int k=0; k<DENSE_3_INPUT; k++) {
                activation += training_inputs[i][k] * dense_3_Weights[k][j];
            }
            dense_3_output[j] = sigmoid(activation);
        }
        // Compute change in dense_3 
        double delta_3[DENSE_3_OUTPUT];
        for (int j=0; j<DENSE_3_OUTPUT; j++) {
            double dError = (training_outputs[i][j] - dense_3_output[j]);
            loss += dError;
            delta_3[j] = dError*dSigmoid(dense_3_output[j]);
        }
        // Compute change in hidden weights
        double delta_2[DENSE_2_OUTPUT];
        for (int j=0; j<DENSE_2_OUTPUT; j++) {
            double dError = 0.0f;
            for(int k=0; k<DENSE_3_OUTPUT; k++) {
                dError+=delta_3[k]*dense_3_Weights[j][k];
            }
            delta_2[j] = dError*dRelu(dense_2_output[j]);
        }
        // Compute change in hidden weights
        double delta_1[DENSE_1_OUTPUT];
        for (int j=0; j<DENSE_1_OUTPUT; j++) {
            double dError = 0.0f;
            for(int k=0; k<DENSE_2_OUTPUT; k++) {
                dError+=delta_2[k]*dense_2_Weights[j][k];
            }
            delta_1[j] = dError*dRelu(dense_1_output[j]);
        }
        // Apply change in dense_3 weights
        for (int j=0; j<DENSE_3_OUTPUT; j++) {
            dense_3_Bias[j] += delta_3[j] * LEARNING_RATE;
            for (int k=0; k<DENSE_2_OUTPUT; k++) {
                dense_3_Weights[k][j] += dense_2_output[k] * delta_3[j] * LEARNING_RATE;
            }
        } 
        // Apply change in dense_2 weights
        for (int j=0; j<DENSE_2_OUTPUT; j++) {
            dense_2_Bias[j] += delta_2[j] * LEARNING_RATE;
            for (int k=0; k<DENSE_1_OUTPUT; k++) {
                dense_2_Weights[k][j] += dense_1_output[k] * delta_2[j] * LEARNING_RATE;
            }
        } 
        // Apply change in dense_1 weights
        for (int j=0; j<DENSE_1_OUTPUT; j++) {
            dense_1_Bias[j] += delta_1[j] * LEARNING_RATE;
            for (int k=0; k<DENSE_1_INPUT; k++) {
                dense_1_Weights[k][j] += training_inputs[i][k] * delta_1[j] * LEARNING_RATE;
            }
        }
    }
    printf("%f\n",loss);
}

int main(void){
    /*
    InitDense1Layer();
    InitDense2Layer();
    InitDense3Layer();
    for (int i=0; i<EPOCHS; i++){
        Training();
    }
    inference();
    */
   
    return 0;
}
