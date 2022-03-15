#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


//void CopyTensor(Tensor* dst, Tensor* src);
//void ReadWeights_binary(const char* fileName, Layer* map);

// global
float tempTensor[2457600];

// struct
typedef struct _Tensor
{
	int height;
	int width;
	int channel;

	float* data;
	//float data[80 * 80 * 96];
}Tensor;

typedef struct _Layer
{
	int inChannel;
	int outChannel;
	int kernel;
	int stride;
	int padding;
	int group;

	int weightSize;

	float* weights;
	float* bias;
}Layer;


void ReadWeights_binary(const char* fileName, Layer* map)
{
	FILE* fp = fopen(fileName, "rb");

	if (fp == NULL)
	{
	}
	float data;

	int inChannel;
	int outChannel;
	int kernel;
	int stride;
	int padding;
	int group;

	float* weights = NULL;
	float* bias = NULL;
	int weightSize;

	int layersNum = -1;
	int weightNum = 0;
	int biasNum = 0;


	while (!feof(fp))
	{
		fread(&data, sizeof(float), 1, fp);
		if ((int)data == 9000)
		{
			//Layer tempLayer = Layer{};

			if (layersNum > -1)
			{
				Layer temp = { inChannel, outChannel, kernel, stride, padding, group, weightSize, weights, bias };
				map[layersNum] = temp;
			}
			//weights.clear();
			//bias.clear();
			weightNum = 0;
			biasNum = 0;
			layersNum += 1;

			fread(&data, sizeof(float), 1, fp);
			inChannel = (int)data;
			fread(&data, sizeof(float), 1, fp);
			outChannel = (int)data;
			fread(&data, sizeof(float), 1, fp);
			kernel = (int)data;
			fread(&data, sizeof(float), 1, fp);
			stride = (int)data;
			fread(&data, sizeof(float), 1, fp);
			padding = (int)data;
			fread(&data, sizeof(float), 1, fp);
			group = (int)data;

			if (group == 1)
			{
				weightSize = inChannel * outChannel * kernel * kernel;
			}
			else
			{
				weightSize = inChannel * 1 * kernel * kernel;
			}

			weights = (float*)malloc(sizeof(float) * weightSize);
			bias = (float*)malloc(sizeof(float) * outChannel);
		}
		else
		{
			if (weightNum < weightSize)
			{
				weights[weightNum] = data;
				//weights.push_back(data);
				weightNum += 1;
			}
			else if (weightNum >= weightSize && biasNum < outChannel)
			{
				bias[biasNum] = data;
				//bias.push_back(data);
				biasNum += 1;
			}
		}
	}
	Layer temp = { inChannel, outChannel, kernel, stride, padding, group, weightSize, weights, bias };
	map[layersNum] = temp;
	fclose(fp);
}











