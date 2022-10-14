#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

double categoryCrossEntropy(double target[], double output[], int size);
void dCrossAndSoftmax(double delta[], double target[], double output[], int size);
#endif