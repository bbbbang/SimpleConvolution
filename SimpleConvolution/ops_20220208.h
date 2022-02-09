#pragma once

#include "utils.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>


std::chrono::microseconds _Relu(Tensor* inputData);
std::chrono::microseconds _ZeroConcat(Tensor* inputData);
std::chrono::microseconds _Softmax(Tensor* inputData);

std::chrono::microseconds _ZeroPadding(Tensor* inputData, int padding);
std::chrono::microseconds _MaxPool(Tensor* inputData, int kernel, int stride, int padding);
std::chrono::microseconds _Resize(Tensor* inputData, float scale);

std::chrono::microseconds _Add(Tensor* inputData, Tensor* outputData);
std::chrono::microseconds _Concat(Tensor* inputData, Tensor* outputData);

std::chrono::microseconds _Relu(Tensor* inputData)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	float* data = inputData->data;
	int size = inputData->width * inputData->height * inputData->channel;

	for (int i = 0; i < size; ++i)
	{
		data[i] = (data[i] < 0) ? 0 : data[i];
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}
std::chrono::microseconds _ZeroConcat(Tensor* inputData)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	float* data = inputData->data;
	int size = inputData->width * inputData->height * inputData->channel;

	for (int i = size; i < size * 2; ++i)
	{
		data[i] = 0;
	}
	inputData->channel = inputData->channel * 2;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}
std::chrono::microseconds _Softmax(Tensor* inputData)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int channel = inputData->channel;
	float sum = 0;
	float* data = inputData->data;

	for (int i = 0; i < channel; ++i)
	{
		float val = std::exp(data[i]);

		data[i] = val;
		sum += val;
	}
	for (int i = 0; i < channel; ++i)
	{
		data[i] = data[i] / sum;
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _ZeroPadding(Tensor* inputData, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int height = inputData->height;
	int width = inputData->width;
	int channel = inputData->channel;
	int area = height * width;

	int outputWidth = (height + padding * 2);
	int outputArea = outputWidth * outputWidth;

	float* data = inputData->data;

	float* tempData = new float[area * channel];
	memcpy(tempData, data, sizeof(float) * area * channel);
	memset(data, 0, sizeof(float) * outputArea * channel);
	float* saveTempPos = tempData;
	float* saveDataPos = data;

	for (int ch = 0; ch < channel; ++ch)
	{
		data += outputWidth;
		for (int row = 0; row < height; ++row)
		{
			++data;
			for (int col = 0; col < width; ++col)
			{
				*data = *tempData;

				++data;
				++tempData;
			}
			++data;
		}
		data += outputWidth;
	}
	inputData->height = outputWidth;
	inputData->width = outputWidth;
	inputData->data = saveDataPos;
	//tempData = saveTempPos;
	delete[] saveTempPos;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}
std::chrono::microseconds _MaxPool(Tensor* inputData, int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int height = inputData->height;
	int width = inputData->width;
	int channel = inputData->channel;
	int area = height * width;
	int tensorSize = area * channel;

	int padHeight = height + padding * 2;
	int padWidth = width + padding * 2;

	int outputHeight = (padHeight - kernel) / stride + 1;
	int outputWidth = (padWidth - kernel) / stride + 1;

	// padding
	if (padding > 0)
	{
		_ZeroPadding(inputData, padding);
	}

	float* data = inputData->data;
	float* tempData = new float[tensorSize];
	memcpy(tempData, data, sizeof(float) * tensorSize);
	//memset(data, 0, sizeof(float) * area * channel);
	float* saveTempPos = tempData;
	float* saveDataPos = data;

	for (int ch = 0; ch < channel; ++ch)
	{
		for (int row = 0; row < height; row += stride)
		{
			for (int col = 0; col < width; col += stride)
			{
				float val1 = *tempData;
				float val2 = *(tempData + 1);
				float val3 = *(tempData + width);
				float val4 = *(tempData + width + 1);

				float m = val1;
				m = val2 > m ? val2 : m;
				m = val3 > m ? val3 : m;
				m = val4 > m ? val4 : m;

				//float max = std::max({ val1, val2, val3, val4 });
				//*data = max;

				//*data = std::max({ val1, val2, val3, val4 });
				*data = m;

				++data;
				tempData += stride;
				//++data;
			}
			tempData += width;
		}
	}

	inputData->height = outputHeight;
	inputData->width = outputWidth;
	inputData->data = saveDataPos;
	//tempData = saveTempPos;
	delete[] saveTempPos;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}
std::chrono::microseconds _Resize(Tensor* inputData, float scale)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int height = inputData->height;
	int width = inputData->width;
	int channel = inputData->channel;
	int area = height * width;

	int outputHeight = height * scale;
	int outputWidth = width * scale;
	int outputArea = outputHeight * outputWidth;

	float* data = inputData->data;
	float* tempData = new float[area * channel];
	memcpy(tempData, data, sizeof(float) * area * channel);
	memset(data, 0, sizeof(float) * outputArea * channel);

	float* saveTempPos = tempData;
	float* saveDataPos = data;

	for (int ch = 0; ch < channel; ++ch)
	{
		for (int row = 0; row < height; ++row)
		{
			for (int col = 0; col < width; ++col)
			{
				float val = *tempData;

				*data = val;
				*(data + 1) = val;
				*(data + outputWidth) = val;
				*(data + outputWidth + 1) = val;

				data += 2;
				++tempData;
			}
			data += outputWidth;
		}
	}

	inputData->height = outputHeight;
	inputData->width = outputWidth;
	inputData->data = saveDataPos;
	tempData = saveTempPos;
	delete[] tempData;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _Add(Tensor* inputData, Tensor* outputData)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	float* data = inputData->data;
	float* _data = outputData->data;

	int size = inputData->width * inputData->height * inputData->channel;

	for (int i = 0; i < size; ++i)
	{
		_data[i] += data[i];
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}
std::chrono::microseconds _Concat(Tensor* inputData, Tensor* outputData)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	float* data = inputData->data;
	float* _data = outputData->data;
	int size = inputData->width * inputData->height * inputData->channel;

	for (int i = size; i < size * 2; ++i)
	{
		data[i] = _data[i - size];
	}

	inputData->channel = inputData->channel * 2;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}



std::chrono::microseconds _Sigmoid(Tensor* tensor);

std::chrono::microseconds _Sigmoid(Tensor* tensor)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	float* data = tensor->data;
	int size = tensor->height * tensor->width * tensor->channel;

	for (int i = 0; i < size; ++i)
	{
		data[i] = 1 / (1 + std::exp(-data[i]));
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}
