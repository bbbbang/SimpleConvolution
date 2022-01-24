#pragma once

#include <chrono>
#include "ops.h"
#include "utils.h"
#include <iostream>


std::chrono::microseconds _Convolution2D_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);
	 
std::chrono::microseconds _Convolution2D_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);
	 
std::chrono::microseconds _Convolution2D_Depthwise_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);
	 
std::chrono::microseconds _Convolution2D_Depthwise_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);
	 
std::chrono::microseconds _Convolution2D_Pointwise_k1_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);


std::chrono::microseconds _Convolution2D_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _Convolution2D_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;
	int area = height * width;
	
	int padHeight = height + padding * 2;
	int padWidth = width + padding * 2;
	int padArea = padWidth * padHeight;

	int outputHeight = (padHeight - kernel) / stride + 1;
	int outputWidth = (padWidth - kernel) / stride + 1;
	int outputArea = outputHeight * outputWidth;

	// padding
	_ZeroPadding(tensor, padding);

	float* data = tensor->data;
	float* tempData = new float[inChannel * padArea];
	memcpy(tempData, data, sizeof(float) * inChannel * padArea);
	memset(data, 0, sizeof(float) * outChannel * outputArea);
	float* saveTempPos = tempData;
	float* saveDataPos = data;

	int kernelSize = inChannel * 9;

	float val_1, val_2, val_3, val_4, val_5, val_6;

	float* tempInputData1 = tempData;
	float* tempInputData2 = tempData + padWidth;
	float* tempInputData3 = tempData + padWidth + padWidth;
	float* tempOutputData = data;

	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		int outKernel = outCh * kernelSize;
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			int kernelIndex = outKernel + inCh * 9;

			float weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
			float weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
			float weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

			for (int row = 0; row < height; row += stride)
			{
				for (int col = 0; col < width; col += stride)
				{
					float val = *data;

					float val1 = *(tempInputData1);
					float val2 = *(tempInputData1 + 1);
					float val3 = *(tempInputData1 + 2);

					float val4 = *(tempInputData2);
					float val5 = *(tempInputData2 + 1);
					float val6 = *(tempInputData2 + 2);

					float val7 = *(tempInputData3);
					float val8 = *(tempInputData3 + 1);
					float val9 = *(tempInputData3 + 2);

					val = val + val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
						val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
						val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9;
					*data = val;

					tempInputData1 += stride;
					tempInputData2 += stride;
					tempInputData3 += stride;
					++data;
				}
				tempInputData1 += padWidth + 2;
				tempInputData2 += padWidth + 2;
				tempInputData3 += padWidth + 2;
			}
			tempInputData1 += padWidth * 2;
			tempInputData2 += padWidth * 2;
			tempInputData3 += padWidth * 2;
			data -= outputArea;
		}
		for (int i = 0; i < outputArea; ++i)
		{
			float val = *data + *bias;
			//val = (val < 0) ? 0 : val;
			*data = val;
			++data;
		}
		++bias;
		tempInputData1 = tempData;
		tempInputData2 = tempData + padWidth;
		tempInputData3 = tempData + padWidth + padWidth;
	}

	tensor->height = outputHeight;
	tensor->width = outputWidth;
	tensor->channel = outChannel;
	tensor->data = saveDataPos;
	tempData = saveTempPos;

	delete[] tempData;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _Convolution2D_Depthwise_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;
	int area = height * width;

	int padHeight = height + padding * 2;
	int padWidth = width + padding * 2;
	int padArea = padWidth * padHeight;

	int outputHeight = (padHeight - kernel) + 1;
	int outputWidth = (padWidth - kernel) + 1;
	int outputArea = outputHeight * outputWidth;

	// padding
	_ZeroPadding(tensor, padding);
	float* data = tensor->data;
	float* tempData = new float[inChannel * padArea];
	memcpy(tempData, data, sizeof(float) * inChannel * padArea);
	memset(data, 0, sizeof(float) * outChannel * outputArea);
	float* saveTempPos = tempData;
	float* saveDataPos = data;

	float val_1, val_2, val_3, val_4, val_5, val_6;

	float* tempInputData1 = tempData;
	float* tempInputData2 = tempData + padWidth;
	float* tempInputData3 = tempData + padWidth + padWidth;
	float* tempOutputData = data;

	for (int inCh = 0; inCh < inChannel; ++inCh)
	{
		int kernelIndex = inCh * 9;

		float weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
		float weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
		float weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

		for (int row = 0; row < height; ++row)
		{
			for (int col = 0; col < width; ++col)
			{
				float val = *data;

				float val1 = *(tempInputData1);
				float val2 = *(tempInputData1 + 1);
				float val3 = *(tempInputData1 + 2);

				float val4 = *(tempInputData2);
				float val5 = *(tempInputData2 + 1);
				float val6 = *(tempInputData2 + 2);

				float val7 = *(tempInputData3);
				float val8 = *(tempInputData3 + 1);
				float val9 = *(tempInputData3 + 2);

				val = val + val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
					val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
					val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9;
				*data = val;

				++tempInputData1;
				++tempInputData2;
				++tempInputData3;
				++data;
			}
			tempInputData1 += 2;
			tempInputData2 += 2;
			tempInputData3 += 2;
		}
		tempInputData1 += padWidth * 2;
		tempInputData2 += padWidth * 2;
		tempInputData3 += padWidth * 2;

		data = tempOutputData;
		for (int i = 0; i < area; ++i)
		{
			float val = *data + *bias;
			//val = (val < 0) ? 0 : val;
			*data = val;

			++data;
		}
		++bias;
		tempOutputData = data;
	}
	tensor->height = outputHeight;
	tensor->width = outputWidth;
	tensor->data = saveDataPos;
	tempData = saveTempPos;
	delete[] tempData;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _Convolution2D_Depthwise_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;
	int area = height * width;
	
	int padHeight = height + padding * 2;
	int padWidth = width + padding * 2;
	int padArea = padWidth * padHeight;

	int outputHeight = (padHeight - kernel) / stride + 1;
	int outputWidth = (padWidth - kernel) / stride + 1;
	int outputArea = outputHeight * outputWidth;

	// padding
	_ZeroPadding(tensor, padding);
	float* data = tensor->data;
	float* tempData = new float[inChannel * padArea];
	memcpy(tempData, data, sizeof(float) * inChannel * padArea);
	memset(data, 0, sizeof(float) * outChannel * outputArea);
	float* saveTempPos = tempData;
	float* saveDataPos = data;

	float val_1, val_2, val_3, val_4, val_5, val_6;

	float* tempInputData1 = tempData;
	float* tempInputData2 = tempData + padWidth;
	float* tempInputData3 = tempData + padWidth + padWidth;
	float* tempOutputData = data;

	for (int inCh = 0; inCh < inChannel; ++inCh)
	{
		int kernelIndex = inCh * 9;

		float weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
		float weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
		float weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

		for (int row = 0; row < height; row += stride)
		{
			for (int col = 0; col < width; col += stride)
			{
				float val = *data;
				
				float val1 = *(tempInputData1);
				float val2 = *(tempInputData1 + 1);
				float val3 = *(tempInputData1 + 2);
				
				float val4 = *(tempInputData2);
				float val5 = *(tempInputData2 + 1);
				float val6 = *(tempInputData2 + 2);
				
				float val7 = *(tempInputData3);
				float val8 = *(tempInputData3 + 1);
				float val9 = *(tempInputData3 + 2);

				val = val + val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
					val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
					val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9;
				*data = val;
				tempInputData1 += stride;
				tempInputData2 += stride;
				tempInputData3 += stride;
				++data;
			}
			tempInputData1 += padWidth + 1;
			tempInputData2 += padWidth + 1;
			tempInputData3 += padWidth + 1;
		}
		tempInputData1 += padWidth + 1;
		tempInputData2 += padWidth + 1;
		tempInputData3 += padWidth + 1;

		data = tempOutputData;
		for (int i = 0; i < area; ++i)
		{
			float val = *data + *bias;
			val = (val < 0) ? 0 : val;
			*data = val;

			++data;
		}
		++bias;
		tempOutputData = data;
	}
	tensor->height = outputHeight;
	tensor->width = outputWidth;
	tensor->data = saveDataPos;
	tempData = saveTempPos;
	delete[] tempData;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _Convolution2D_Pointwise_k1_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;
	int area = height * width;

	float* data = tensor->data;
	float* tempData = new float[inChannel * area];
	memcpy(tempData, data, sizeof(float) * inChannel * area);
	memset(data, 0, sizeof(float) * outChannel * area);
	float* saveTempPos = tempData;
	float* saveDataPos = data;

	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			float weightVal = *weight;
			int inPos = inCh * area;

			for (int i = 0; i < area; ++i)
			{
				float o = *data;
				o = o + tempData[inPos + i] * weightVal;
				(*data) = o;
				++data;
			}
			++weight;
			data -= area;
		}

		for (int i = 0; i < area; ++i)
		{
			float val = *data + *bias;
			//val = (val < 0) ? 0 : val;
			*data = val;
			++data;
		}
		++bias;
	}
	tensor->channel = outChannel;
	tensor->data = saveDataPos;
	tempData = saveTempPos;
	delete[] tempData;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}




void Postprocessing(Tensor* offset, Tensor* size, Tensor* keypoint);

void Postprocessing(Tensor* offset, Tensor* size, Tensor* keypoint)
{

}
