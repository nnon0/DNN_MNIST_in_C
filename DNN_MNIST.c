#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUTNO		784
#define HIDDENNO	32
#define OUTPUTNO	10
#define WEIGHT1_PATH	"./data/W1.txt"
#define WEIGHT2_PATH	"./data/W2.txt"
#define BIAS1_PATH		"./data/b1.txt"
#define BIAS2_PATH		"./data/b2.txt"

void softmax(double node[OUTPUTNO], double y[OUTPUTNO]);
void weight1(double w[HIDDENNO][INPUTNO]);
void weight2(double w[OUTPUTNO][HIDDENNO]);
void bias1(double b[HIDDENNO]);
void bias2(double b[OUTPUTNO]);
void forward(double weight_1[HIDDENNO][INPUTNO], double weight_2[OUTPUTNO][HIDDENNO], double bias_1[HIDDENNO], double bias_2[OUTPUTNO], double input_data[INPUTNO], double output_data[OUTPUTNO]);
double getdata(double input_data[INPUTNO]);
double relu(double input_data);


double w1[HIDDENNO][INPUTNO] = { 0, };
double w2[OUTPUTNO][HIDDENNO] = { 0, };
double b1[HIDDENNO] = { 0, };
double b2[OUTPUTNO] = { 0, };

int main() {
	double label = 2;
	double input[784] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,125,171,255,255,150,93,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,169,253,253,253,253,253,253,218,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,169,253,253,253,213,142,176,253,253,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,52,250,253,210,32,12,0,6,206,253,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,251,210,25,0,0,0,122,248,253,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,18,0,0,0,0,209,253,253,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,117,247,253,198,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,76,247,253,231,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,253,253,144,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,176,246,253,159,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,234,253,233,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,198,253,253,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,78,248,253,189,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,200,253,253,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,134,253,253,173,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,248,253,253,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,248,253,253,43,20,20,20,20,5,0,5,20,20,37,150,150,150,147,10,0,0,0,0,0,0,0,0,0,248,253,253,253,253,253,253,253,168,143,166,253,253,253,253,253,253,253,123,0,0,0,0,0,0,0,0,0,174,253,253,253,253,253,253,253,253,253,253,253,249,247,247,169,117,117,57,0,0,0,0,0,0,0,0,0,0,118,123,123,123,166,253,253,253,155,123,123,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	double hidden[32] = { 0, };
	double output[10] = { 0, };
	
	for (int i = 0; i < INPUTNO; i++)
		input[i] = input[i] / 255;

	weight1(w1);	
	weight2(w2);
	bias1(b1);
	bias2(b2);

	//getdata(input);
	forward(w1, w2, b1, b2, input, output); // return 출력값을 확인 가능

	for (int i = 0; i < OUTPUTNO; i++)
		printf("output[%d] : %.9f \n", i, output[i]);
	
	printf("Label : %.1f", label);

	return 0;
}

void weight1(double w[HIDDENNO][INPUTNO]) {
	FILE* file = NULL;
	fopen_s(&file, WEIGHT1_PATH, "r");

	
	if (file == NULL) {
		perror("open %s failed!\n", WEIGHT1_PATH);
		return;
	}

	char *temp1 = NULL, *temp2 = NULL;
	double num = 0;
	int i = 0, j = 0;

	while (!feof(file)) {
		char buf[4096] = { NULL, };
		fgets(buf, 4096, file);
		temp1 = strtok_s(buf, " ", &temp2);
		while (temp1 != NULL) {
			num = atof(temp1); // 문자열을 실수형으로 변경
			if (i == 32) {
				i = 0;
				j++;
			}
			if (num == 0) break;
			//printf("i : %d, j: %d\n", i, j);
			w[i][j] = num;
			i++;
			temp1 = strtok_s(NULL, " ", &temp2);			
		}
	}
	fclose(file);
}

void weight2(double w[OUTPUTNO][HIDDENNO]) {
	FILE* file = NULL;
	fopen_s(&file, WEIGHT2_PATH, "r");


	if (file == NULL) {
		perror("open %s failed!\n", WEIGHT2_PATH);
		return;
	}

	char* temp1 = NULL, * temp2 = NULL;
	double num = 0;
	int i = 0, j = 0;

	while (!feof(file)) {
		char buf[4096] = { NULL, };
		fgets(buf, 4096, file);
		temp1 = strtok_s(buf, " ", &temp2);
		while (temp1 != NULL) {
			num = atof(temp1); // 문자열을 실수형으로 변경
			if (i == 10) {
				i = 0;
				j++;
			}
			if (num == 0) break;
			w[i][j] = num;
			i++;
			temp1 = strtok_s(NULL, " ", &temp2);
		}
	}

	fclose(file);
}

void bias1(double b[HIDDENNO]) {
	FILE* file = NULL;
	fopen_s(&file, BIAS1_PATH, "r");


	if (file == NULL) {
		perror("open %s failed!\n", BIAS1_PATH);
		return;
	}

	char* temp1 = NULL, * temp2 = NULL;
	double num = 0;
	int i = 0;

	while (!feof(file)) {
		char buf[4096] = { NULL, };
		fgets(buf, 4096, file);
		temp1 = strtok_s(buf, " ", &temp2);
		while (temp1 != NULL) {
			num = atof(temp1); // 문자열을 실수형으로 변경
			b[i] = num;
			i++;
			temp1 = strtok_s(NULL, " ", &temp2);
		}
	}

	fclose(file);
}

void bias2(double b[OUTPUTNO]) {
	FILE* file = NULL;
	fopen_s(&file, BIAS2_PATH, "r");


	if (file == NULL) {
		perror("open %s failed!\n", BIAS2_PATH);
		return;
	}

	char* temp1 = NULL, * temp2 = NULL;
	double num = 0;
	int i = 0;

	while (!feof(file)) {
		char buf[4096] = { NULL, };
		fgets(buf, 4096, file);
		temp1 = strtok_s(buf, " ", &temp2);
		while (temp1 != NULL) {
			num = atof(temp1); // 문자열을 실수형으로 변경
			b[i] = num;
			i++;
			temp1 = strtok_s(NULL, " ", &temp2);
		}
	}
	fclose(file);
}

void forward(double weight_1[HIDDENNO][INPUTNO], double weight_2[OUTPUTNO][HIDDENNO], double bias_1[HIDDENNO], double bias_2[OUTPUTNO], double input_data[INPUTNO], double output_data[OUTPUTNO]) {
	double temp_hidden[HIDDENNO] = { 0, };
	double temp_output[OUTPUTNO] = { 0, };

	for(int i = 0; i<HIDDENNO;i++){
		for (int j = 0; j < INPUTNO; j++) {
			temp_hidden[i] += (input_data[j]*weight_1[i][j]);
		}
		temp_hidden[i] = relu((temp_hidden[i] + bias_1[i]));
	}

	for (int i = 0; i < OUTPUTNO; i++) {
		for (int j = 0; j < HIDDENNO; j++) {
			temp_output[i] += (temp_hidden[j] * weight_2[i][j]);
		}
		temp_hidden[i] += bias_2[i];
	}

	softmax(temp_output, output_data);
}

double getdata(double input_data[INPUTNO]) {}

double relu(double input_data) {
	if (input_data > 0)
		return input_data;
	else
		return 0;
}

void softmax(double node[OUTPUTNO], double y[OUTPUTNO]) {
	double sum_exp_node = 0;

	for (int i = 0; i < OUTPUTNO; i++) {
		sum_exp_node += exp(node[i]);
	}

	for (int i = 0; i < OUTPUTNO; i++) {
		y[i] = exp(node[i]) / sum_exp_node;
	}	
}
