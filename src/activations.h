#ifndef ACTIVATION_H
#define ACTIVATION_H

double sigmoid(double x);
double dSigmoid(double x);
double relu(double x);
double dRelu(double x);
double softmax(double arr[], int size);
void dSoftmax(double delta[], double output[], int size, double totalExponential);
#endif