#include "../src/libs.h"
#include "../src/activations.h"
#include "../src/lossFunctions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define N 5000
#define LEARNING_RATE 0.1
#define EPOCHS 10000

#define DENSE_1_INPUT 768
#define DENSE_1_OUTPUT 256
#define DENSE_2_INPUT 256
#define DENSE_2_OUTPUT 64
#define DENSE_3_INPUT 64
#define DENSE_3_OUTPUT 10
#define numTrainingSets 60000
#define PRINT_PERIOD 5000

double dense_1_Bias[DENSE_1_OUTPUT];
double dense_2_Bias[DENSE_2_OUTPUT];
double dense_3_Bias[DENSE_3_OUTPUT];
double dense_1_Weights[DENSE_1_INPUT][DENSE_1_OUTPUT];
double dense_2_Weights[DENSE_2_INPUT][DENSE_2_OUTPUT];
double dense_3_Weights[DENSE_3_INPUT][DENSE_3_OUTPUT];
double dense_1_output[DENSE_1_OUTPUT];
double dense_2_output[DENSE_2_OUTPUT];
double dense_3_output[DENSE_3_OUTPUT];

double training_inputs[numTrainingSets][DENSE_1_INPUT];
double training_outputs[numTrainingSets][DENSE_3_OUTPUT];
int trainingSetOrder[numTrainingSets];

double deltaCrossAndSoftmax[DENSE_3_OUTPUT];
double delta_2[DENSE_2_OUTPUT];
double delta_1[DENSE_1_OUTPUT];
    

//載入訓練集
void ReadMnistData(){
    FILE *fp;
    char str[N + 1];
    if ( (fp = fopen("networks/label.txt", "rt")) == NULL ) {
        puts("Fail to open file!");
        exit(0);
    }
    int i =0;
    while( fgets(str, N, fp) != NULL ) {
        const char* d = ",";
        char *p;
        p = strtok(str, d);
        int j =0;
        while (p != NULL) {
            sscanf(p, "%lf", &training_outputs[i][j]);
            p = strtok(NULL, d);		   
            j++;
        }
        i++;
    }
    fclose(fp);
    if ( (fp = fopen("networks/minist.txt", "rt")) == NULL ) {
        puts("Fail to open file!");
        exit(0);
    }
    i =0;
    while( fgets(str, N, fp) != NULL ) {
        const char* d = ",";
        char *p;
        p = strtok(str, d);
        int j =0;
        while (p != NULL) {
            double temp;
            sscanf(p, "%lf", &temp);
            training_inputs[i][j] = temp/255;
            p = strtok(NULL, d);		   
            j++;
        }
        i++;
    }
    fclose(fp);
}

//印出one hot label
void ShowLabel(){
    for (int i=0;i<numTrainingSets;i++){
        for (int j=0;j<DENSE_3_OUTPUT;j++){
            printf("%f,",training_outputs[i][j]);
        }
        printf("\n");
    }
}

//印出mnist
void ShowMnist(){
    for (int i=0;i<numTrainingSets;i++){
        for (int j=0;j<DENSE_1_INPUT;j++){
            printf("%f,",training_inputs[i][j]);
        }
        printf("\n");
    }
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
    //shuffle(trainingSetOrder,numTrainingSets);
    //for (int i=0;i<60000;i++)printf("%d\n",trainingSetOrder[i]);
    for (int x=0; x<numTrainingSets; x++) {
        //int i = trainingSetOrder[x];
        //printf("%dtest%d\n",x,i);
        double activation;
        for (int j=0; j<DENSE_1_OUTPUT; j++) {
            activation=dense_1_Bias[j];
            for (int k=0; k<DENSE_1_INPUT; k++) activation += training_inputs[x][k] * dense_1_Weights[k][j];
            //for (int k=0; k<DENSE_1_INPUT; k++) activation += training_inputs[i][k];
            dense_1_output[j] = relu(activation);
        }
        //for (int z=0; z <DENSE_1_OUTPUT; z++) printf("%f,",dense_1_output[z]);
        for (int j=0; j<DENSE_2_OUTPUT; j++) {
            double activation=dense_2_Bias[j];
            for (int k=0; k<DENSE_2_INPUT; k++) {
                activation += dense_1_output[k] * dense_2_Weights[k][j];
            }
            dense_2_output[j] = relu(activation);
        }
        //for (int z=0; z <DENSE_2_OUTPUT; z++) printf("%f,",dense_2_output[z]);
        for (int j=0; j<DENSE_3_OUTPUT; j++) {
            double activation=dense_3_Bias[j];
            for (int k=0; k<DENSE_3_INPUT; k++) {
                activation += dense_2_output[k] * dense_3_Weights[k][j];
            }
            dense_3_output[j] = activation;
        }
        //for (int z=0; z <DENSE_3_OUTPUT; z++) printf("%f,",dense_3_output[z]);
        double totalExponential = softmax(dense_3_output, DENSE_3_OUTPUT);
        //for (int z=0; z <DENSE_3_OUTPUT; z++) printf("%f,",dense_3_output[z]);
        double loss = categoryCrossEntropy(training_outputs[x], dense_3_output, DENSE_3_OUTPUT);
        if (x % PRINT_PERIOD == 0)printf("%f\n",loss);
        dCrossAndSoftmax(deltaCrossAndSoftmax, training_outputs[x], dense_3_output, DENSE_3_OUTPUT);
        //for (int z=0; z <DENSE_3_OUTPUT; z++) printf("%f,",deltaCrossAndSoftmax[z]);
        for (int z=0; z <DENSE_3_OUTPUT; z++) {
            deltaCrossAndSoftmax[z] = loss * deltaCrossAndSoftmax[z];
        }
        //for (int z=0; z <DENSE_3_OUTPUT; z++) printf("%f,",deltaCrossAndSoftmax[z]);
        for (int j=0; j<DENSE_2_OUTPUT; j++) {
            double dError = 0.0f;
            for(int k=0; k<DENSE_3_OUTPUT; k++) {
                dError+=deltaCrossAndSoftmax[k]*dense_3_Weights[j][k];
            }
            delta_2[j] = dError*dRelu(dense_2_output[j]);
        }
        for (int j=0; j<DENSE_1_OUTPUT; j++) {
            double dError = 0.0f;
            for(int k=0; k<DENSE_2_OUTPUT; k++) {
                dError+=delta_2[k]*dense_2_Weights[j][k];
            }
            delta_1[j] = dError*dRelu(dense_1_output[j]);
        }
        // Apply change in dense_3 weights
        for (int j=0; j<DENSE_3_OUTPUT; j++) {
            dense_3_Bias[j] += deltaCrossAndSoftmax[j] * LEARNING_RATE;
            for (int k=0; k<DENSE_2_OUTPUT; k++) {
                dense_3_Weights[k][j] += dense_2_output[k] * deltaCrossAndSoftmax[j] * LEARNING_RATE;
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
                dense_1_Weights[k][j] += training_inputs[x][k] * delta_1[j] * LEARNING_RATE;
            }
        }
    }
}

int main(void){
    ReadMnistData();
    InitOrder(trainingSetOrder, numTrainingSets);
    InitDense1Layer();
    InitDense2Layer();
    InitDense3Layer();
    Training();
    /*
    for (int i=0; i<EPOCHS; i++){
        Training();
    }
    inference();
    */ 
    return 0;
}
