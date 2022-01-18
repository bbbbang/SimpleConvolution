#pragma once
#include <immintrin.h>

#include "ops.h"

#include <iostream>
//// all operations are vectorized by 10, since input image size is 160


//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// normal convolution ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::chrono::microseconds Convolution2D_k3_s1(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding);

template <typename T>
std::chrono::microseconds Convolution2D_k3_s2(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding);


// yet no need to implement
template <typename T>
std::chrono::microseconds Convolution2D_k3_s1(T* inputData, T* outputData, T* weight, T* bias,
											int height, int width, int inChannel, int outChannel,
											int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}


template <typename T>
std::chrono::microseconds Convolution2D_k3_s2(T* inputData, T* outputData, T* weight, T* bias,
											int height, int width, int inChannel, int outChannel,
											int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int area = height * width;

	int padHeight = height + padding * 2;
	int padWidth = width + padding * 2;
	int padArea = padWidth * padHeight;

	int outputHeight = (padHeight - kernel) / stride + 1;
	int outputWidth = (padWidth - kernel) / stride + 1;
	int outputArea = outputHeight * outputWidth;

	#pragma region padding

	// make padding tensor
	float* padInput = new float[inChannel * padHeight * padWidth];
	for (int i = 0; i < inChannel * padHeight * padWidth; ++i)
	{
		padInput[i] = 0;
	}

	ZeroPadding(inputData, padInput, height, width, inChannel, padding);

	#pragma endregion

	int kernelSize = inChannel * 9;
	int vectorizeCount = 10 * stride;
	int repeat = padWidth / vectorizeCount;
	repeat *= repeat;

	int tempTopInputPos;
	int tempMidInputPos;
	int tempBotInputPos;
	int tempOutputPos;

	//T val_1, val_2, val_3, val_4, val_5, val_6;

	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		int outChIndex = outCh * outputArea;
		int kernelArea = outCh * kernelSize;
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			int inChIndex = inCh * padArea;
			int kernelIndex = kernelArea + inCh * 9;

			T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
			T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
			T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];


			#pragma region vectorize

			int i = 0;
			int j = 0;
			repeat = padWidth / vectorizeCount;
			repeat *= repeat;
			while (repeat-- > 0)
			{
				tempTopInputPos = inChIndex + j * padWidth + i;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = outChIndex + j / stride * outputWidth + i / stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;



				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
				tempTopInputPos += stride;
				tempMidInputPos += stride;
				tempBotInputPos += stride;

				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;

				i += vectorizeCount;
				if (i >= width)
				{
					i = 0;
					j += vectorizeCount;
				}
			}

			#pragma endregion
		}
		//for (int r = 0; r < outputHeight; ++r)
		//{
		//	for (int c = 0; c < outputWidth; ++c)
		//	{
		//		T val = outputData[outCh * outputArea + r * outputWidth + c] + bias[outCh];
		//		//outputData[outCh * outputArea + r * outputWidth + c] = val;
		//		outputData[outCh * outputArea + r * outputWidth + c] = (val < 0) ? 0 : val;
		//	}
		//}
	}

	delete[]padInput;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}



//template <typename T>
//std::chrono::microseconds Convolution2D_k3_s2(T* inputData, T* outputData, T* weight, T* bias,
//	int height, int width, int inChannel, int outChannel,
//	int kernel, int stride, int padding)
//{
//	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
//
//	int area = height * width;
//
//	int padHeight = height + padding * 2;
//	int padWidth = width + padding * 2;
//	int padArea = padWidth * padHeight;
//
//	int outputHeight = (padHeight - kernel) / stride + 1;
//	int outputWidth = (padWidth - kernel) / stride + 1;
//	int outputArea = outputHeight * outputWidth;
//
//	// padding
//	float* padInput = new float[inChannel * padArea];
//	for (int i = 0; i < inChannel * padArea; ++i)
//	{
//		padInput[i] = 0;
//	}
//
//	ZeroPadding(inputData, padInput, height, width, inChannel, padding);
//
//	int kernelSize = inChannel * 9;
//	int vectorizeCount = 2 * stride;
//	int repeat = padWidth / vectorizeCount;
//	repeat *= repeat;
//
//	int tempTopInputPos;
//	int tempMidInputPos;
//	int tempBotInputPos;
//	int tempOutputPos;
//
//	T val_1, val_2, val_3, val_4, val_5, val_6;
//
//	for (int outCh = 0; outCh < outChannel; ++outCh)
//	{
//		int outChIndex = outCh * padArea;
//		int kernelArea = outCh * kernelSize;
//		for (int inCh = 0; inCh < inChannel; ++inCh)
//		{
//			int inChIndex = inCh * padArea;
//			int kernelIndex = kernelArea + inCh * 9;
//
//			T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
//			T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
//			T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];
//
//			int i = 0;
//			int j = 0;
//			while (repeat-- > 0)
//			{
//				tempTopInputPos = inChIndex + j * padWidth + i;
//				tempMidInputPos = tempTopInputPos + padWidth;
//				tempBotInputPos = tempMidInputPos + padWidth;
//				tempOutputPos = outChIndex + j / stride * outputWidth + i / stride;
//
//				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//				val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//				outputData[tempOutputPos] += val_1;
//				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//				val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//				outputData[tempOutputPos] += val_2;
//				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//				val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//				outputData[tempOutputPos++] += val_3;
//				tempTopInputPos += stride;
//				tempMidInputPos += stride;
//				tempBotInputPos += stride;
//
//				outputData[tempOutputPos] += val_1 + val_2 + val_3;
//				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//				val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//				outputData[tempOutputPos] += val_1;
//				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//				val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//				outputData[tempOutputPos] += val_2;
//				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//				val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//				outputData[tempOutputPos++] += val_3;
//				tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount + stride;
//				tempMidInputPos = tempTopInputPos + padWidth;
//				tempBotInputPos = tempMidInputPos + padWidth;
//				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//
//
//
//				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//				val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//				outputData[tempOutputPos] += val_1;
//				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//				val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//				outputData[tempOutputPos] += val_2;
//				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//				val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//				outputData[tempOutputPos++] += val_3;
//				tempTopInputPos += stride;
//				tempMidInputPos += stride;
//				tempBotInputPos += stride;
//
//
//				outputData[tempOutputPos] += val_1 + val_2 + val_3;
//				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//				val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//				outputData[tempOutputPos] += val_1;
//				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//				val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//				outputData[tempOutputPos] += val_2;
//				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//				val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//				outputData[tempOutputPos++] += val_3;
//
//				i += vectorizeCount;
//
//				if (i > padWidth)
//				{
//					i = 0;
//					j += vectorizeCount;
//				}
//			}
//		}
//		for (int r = 0; r < outputHeight; ++r)
//		{
//			for (int c = 0; c < outputWidth; ++c)
//			{
//				T val = outputData[outCh * outputArea + r * outputWidth + c] + bias[outCh];
//				outputData[outCh * outputArea + r * outputWidth + c] = val;
//				//outputData[outCh * outputArea + r * outputWidth + c] = (val < 0) ? 0 : val;
//			}
//		}
//	}
//
//	delete[]padInput;
//
//	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
//	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//}
//
//
//






//////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// depthwise convolution /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::chrono::microseconds Convolution2D_Depthwise_k3_s1(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding);

template <typename T>
std::chrono::microseconds Convolution2D_Depthwise_k3_s2(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding);



template <typename T>
std::chrono::microseconds Convolution2D_Depthwise_k3_s1(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int area = height * width;

	int padHeight = height + padding * 2;
	int padWidth = width + padding * 2;
	int padArea = padWidth * padHeight;

	int outputHeight = (padHeight - kernel) + 1;
	int outputWidth = (padWidth - kernel) + 1;
	int outputArea = outputHeight * outputWidth;

	// padding
	float* padInput = new float[inChannel * padArea];
	for (int i = 0; i < inChannel * padArea; ++i)
	{
		padInput[i] = 0;
	}

	ZeroPadding(inputData, padInput, height, width, inChannel, padding);

	int vectorizeCount = 10;
	int repeat = padWidth / vectorizeCount;
	repeat *= repeat;

	int tempTopInputPos;
	int tempMidInputPos;
	int tempBotInputPos;
	int tempOutputPos;

	T val_1, val_2, val_3, val_4, val_5, val_6;

	for (int inCh = 0; inCh < inChannel; ++inCh)
	{
		int inChIndex = inCh * padArea;
		int kernelIndex = inCh * 9;

		T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
		T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
		T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

		int i = 0;
		int j = 0;
		repeat = padWidth / vectorizeCount;
		repeat *= repeat;
		while (repeat-- > 0)
		{
			tempTopInputPos = inChIndex + j * padWidth + i;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = inChIndex + j / stride * outputWidth + i / stride;

			#pragma region vectorized
			// row 1
			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


			// row 2
			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 3
			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 4
			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 5
			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 6
			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 7
			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			++tempTopInputPos;
			++tempMidInputPos;
			++tempBotInputPos;

			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
			#pragma endregion

			i += vectorizeCount;
			if (i >= width)
			{
				i = 0;
				j += vectorizeCount;
			}
		}

		for (int r = 0; r < outputHeight; ++r)
		{
			for (int c = 0; c < outputWidth; ++c)
			{
				T val = outputData[inCh * outputArea + r * outputWidth + c] + bias[inCh];
				outputData[inCh * outputArea + r * outputWidth + c] = (val < 0) ? 0 : val;
			}
		}
	}
	delete[]padInput;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}


template <typename T>
std::chrono::microseconds Convolution2D_Depthwise_k3_s2(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int area = height * width;

	int padHeight = height + padding * 2;
	int padWidth = width + padding * 2;
	int padArea = padWidth * padHeight;

	int outputHeight = (padHeight - kernel) / stride + 1;
	int outputWidth = (padWidth - kernel) / stride + 1;
	int outputArea = outputHeight * outputWidth;

	// padding
	float* padInput = new float[inChannel * padArea];
	for (int i = 0; i < inChannel * padArea; ++i)
	{
		padInput[i] = 0;
	}

	ZeroPadding(inputData, padInput, height, width, inChannel, padding);

	int vectorizeCount = 10 * stride;
	int repeat = padWidth / vectorizeCount;
	repeat *= repeat;

	int tempTopInputPos;
	int tempMidInputPos;
	int tempBotInputPos;
	int tempOutputPos;

	T val_1;
	T val_2;
	T val_3;

	int count = 0;

	for (int inCh = 0; inCh < inChannel; ++inCh)
	{
		int inChIndex = inCh * padArea;
		int outChIndex = inCh * outputArea;
		int kernelIndex = inCh * 9;

		T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
		T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
		T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

		int i = 0;
		int j = 0;
		repeat = padWidth / vectorizeCount;
		repeat *= repeat;
		while (repeat-- > 0)
		{
			//count++;
			tempTopInputPos = inChIndex + j * padWidth + i;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = outChIndex + j / stride * outputWidth + i / stride;

			#pragma region vectorized

			// row 1
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 2
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 3
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 4
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 5
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 6
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 7
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 8
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 9
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;

			// row 10
			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;
			tempTopInputPos += stride;
			tempMidInputPos += stride;
			tempBotInputPos += stride;

			outputData[tempOutputPos] += val_1 + val_2 + val_3;
			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
			outputData[tempOutputPos] += val_1;
			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
			outputData[tempOutputPos] += val_2;
			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
			outputData[tempOutputPos++] += val_3;

			#pragma endregion

			i += vectorizeCount;
			if (i >= width)
			{
				i = 0;
				j += vectorizeCount;
			}
		}

		for (int r = 0; r < outputHeight; ++r)
		{
			for (int c = 0; c < outputWidth; ++c)
			{
				T val = outputData[inCh * outputArea + r * outputWidth + c] + bias[inCh];
				outputData[inCh * outputArea + r * outputWidth + c] = (val < 0) ? 0 : val;
			}
		}
	}
	delete[]padInput;
	//std::cout << "depthwise : " << count << std::endl;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}















//////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// pointwise convolution /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::chrono::microseconds Convolution2D_Pointwise_k1_s1(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding);

// yet no need to implement
template <typename T>
std::chrono::microseconds Convolution2D_Pointwise_k1_s2(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding);




//template <typename T>
//std::chrono::microseconds Convolution2D_Pointwise_k1_s1(T* inputData, T* outputData, T* weight, T* bias,
//	int height, int width, int inChannel, int outChannel,
//	int kernel, int stride, int padding)
//{
//	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
//
//	int area = height * width;
//
//	int rowIndex;
//	int tempInputPos;
//	int tempOutputPos;
//
//	if (height % 10 == 0)
//	{
//		int vectorizeCount = 10;
//		int repeat = height / vectorizeCount;
//		repeat *= repeat;
//
//		for (int outCh = 0; outCh < outChannel; ++outCh)
//		{
//			int outChIndex = outCh * area;
//			for (int inCh = 0; inCh < inChannel; ++inCh)
//			{
//				int inChIndex = inCh * area;
//				T weightVal = *weight;
//
//				int i = 0;
//				int j = 0;
//				repeat = height / vectorizeCount;
//				repeat *= repeat;
//				while (repeat-- > 0)
//				{
//					rowIndex = j * width;
//					tempInputPos = inChIndex + rowIndex + i;
//					tempOutputPos = outChIndex + rowIndex + i;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					rowIndex += width;
//					tempInputPos += width;
//					tempOutputPos += width;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					rowIndex += width;
//					tempInputPos += width;
//					tempOutputPos += width;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					rowIndex += width;
//					tempInputPos += width;
//					tempOutputPos += width;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					rowIndex += width;
//					tempInputPos += width;
//					tempOutputPos += width;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					rowIndex += width;
//					tempInputPos += width;
//					tempOutputPos += width;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					rowIndex += width;
//					tempInputPos += width;
//					tempOutputPos += width;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					rowIndex += width;
//					tempInputPos += width;
//					tempOutputPos += width;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					rowIndex += width;
//					tempInputPos += width;
//					tempOutputPos += width;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					rowIndex += width;
//					tempInputPos += width;
//					tempOutputPos += width;
//					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
//					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
//					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
//					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
//					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
//					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
//					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
//					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
//					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;
//
//					i += vectorizeCount;
//					if (i > width)
//					{
//						i = 0;
//						j += vectorizeCount;
//					}
//				}
//				++weight;
//			}
//			T* v = outputData + outCh * area - 1;
//			T biasVal = bias[outCh];
//			for (int r = 0; r < area; ++r)
//			{
//				T val = *++v + biasVal;
//				*v = (val < 0) ? 0 : val;
//			}
//		}
//	}
//	else
//	{
//		for (int outCh = 0; outCh < outChannel; ++outCh)
//		{
//			int outChIndex = outCh * area;
//			for (int inCh = 0; inCh < inChannel; ++inCh)
//			{
//				int inChIndex = inCh * area;
//				T weightVal = *weight;
//				for (int row = 0; row < height; ++row)
//				{
//					int rowIndex = row * width;
//					for (int col = 0; col < width; ++col)
//					{
//						int outputPos = outChIndex + rowIndex + col;
//						outputData[outputPos] += inputData[inChIndex + rowIndex + col] * weightVal;
//					}
//				}
//				++weight;
//			}
//			T* v = outputData + outCh * area - 1;
//			T biasVal = bias[outCh];
//			for (int r = 0; r < area; ++r)
//			{
//				T val = *++v + biasVal;
//				*v = (val < 0) ? 0 : val;
//			}
//		}
//	}
//
//
//	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
//	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//}
//



//template <typename T>
//std::chrono::microseconds Convolution2D_Pointwise_k1_s1(T* inputData, T* outputData, T* weight, T* bias,
//	int height, int width, int inChannel, int outChannel,
//	int kernel, int stride, int padding)
//{
//	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
//
//	int area = height * width;
//
//	int rowIndex;
//	int tempInputPos;
//	int tempOutputPos;
//
//	if (height % 10 == 0)
//	{
//		int vectorizeCount = 10;
//		int repeat = height / vectorizeCount;
//		repeat *= repeat;
//
//		for (int outCh = 0; outCh < outChannel; ++outCh)
//		{
//			int outChIndex = outCh * area;
//			for (int inCh = 0; inCh < inChannel; ++inCh)
//			{
//				int inChIndex = inCh * area;
//				T weightVal = *weight;
//
//				int i = 0;
//				int j = 0;
//				repeat = height / vectorizeCount;
//				repeat *= repeat;
//				while (repeat-- > 0)
//				{
//					rowIndex = j * width;
//					tempInputPos = inChIndex + rowIndex + i;
//					tempOutputPos = outChIndex + rowIndex + i;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					tempInputPos += width - vectorizeCount;
//					tempOutputPos += width - vectorizeCount;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					tempInputPos += width - vectorizeCount;
//					tempOutputPos += width - vectorizeCount;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					tempInputPos += width - vectorizeCount;
//					tempOutputPos += width - vectorizeCount;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					tempInputPos += width - vectorizeCount;
//					tempOutputPos += width - vectorizeCount;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					tempInputPos += width - vectorizeCount;
//					tempOutputPos += width - vectorizeCount;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					tempInputPos += width - vectorizeCount;
//					tempOutputPos += width - vectorizeCount;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					tempInputPos += width - vectorizeCount;
//					tempOutputPos += width - vectorizeCount;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					tempInputPos += width - vectorizeCount;
//					tempOutputPos += width - vectorizeCount;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					tempInputPos += width - vectorizeCount;
//					tempOutputPos += width - vectorizeCount;
//					outputData[tempOutputPos++] += (inputData[tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos] +
//						inputData[++tempInputPos] + inputData[++tempInputPos] + inputData[++tempInputPos]) * weightVal;
//					//outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//					//outputData[++tempOutputPos] += inputData[++tempInputPos] * weightVal;
//
//					i += vectorizeCount;
//					if (i >= width)
//					{
//						i = 0;
//						j += vectorizeCount;
//					}
//				}
//				++weight;
//			}
//			T* v = outputData + outCh * area - 1;
//			T biasVal = bias[outCh];
//			for (int r = 0; r < area; ++r)
//			{
//				T val = *++v + biasVal;
//				*v = (val < 0) ? 0 : val;
//			}
//		}
//	}
//	else
//	{
//		for (int outCh = 0; outCh < outChannel; ++outCh)
//		{
//			int outChIndex = outCh * area;
//			for (int inCh = 0; inCh < inChannel; ++inCh)
//			{
//				int inChIndex = inCh * area;
//				T weightVal = *weight;
//				for (int row = 0; row < height; ++row)
//				{
//					int rowIndex = row * width;
//					for (int col = 0; col < width; ++col)
//					{
//						int outputPos = outChIndex + rowIndex + col;
//						outputData[outputPos] += inputData[inChIndex + rowIndex + col] * weightVal;
//					}
//				}
//				++weight;
//			}
//			T* v = outputData + outCh * area - 1;
//			T biasVal = bias[outCh];
//			for (int r = 0; r < area; ++r)
//			{
//				T val = *++v + biasVal;
//				*v = (val < 0) ? 0 : val;
//			}
//		}
//	}
//
//
//	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
//	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//}
//


template <typename T>
std::chrono::microseconds Convolution2D_Pointwise_k1_s1(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int area = height * width;

	int rowIndex = 0;
	int tempInputPos;
	int tempOutputPos;

	int vectorizeCount = 10;
	int repeat = height / vectorizeCount;
	repeat *= repeat;

	int outChIndex, inChIndex;

	T* tempOutputData = outputData;
	T* tempInputData = inputData;

	#pragma omp parallel
	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		outChIndex = outCh * area;
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			inChIndex = inCh * area;
			T weightVal = *weight;

			//for (int row = 0; row < height; ++row)
			//{
			//	int t = inChIndex + row * width;
			//	int o = outChIndex + row * width;
			//	for (int col = 0; col < width; ++col)
			//	{
			//		int tempInputPos = t + col;
			//		int tempOutputPos = o + col;

			//		outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
			//	}
			//}

			#pragma omp for
			for (int row = 0; row < height; ++row)
			{
				int t = inChIndex + row * width;
				int o = outChIndex + row * width;
				for (int col = 0; col < width; ++col)
				{
					int tempInputPos = t + col;
					int tempOutputPos = o + col;

					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
				}
			}


			//for (int row = 0; row < height; ++row)
			//{
			//	int i = 0;
			//	repeat = height / vectorizeCount;

			//	int t = outChIndex + row * width;
			//	int tt = inChIndex + row * width;
			//	while (repeat-- > 0)
			//	{ 
			//		int a = t + i;
			//		int b = tt + i;
			//		outputData[a + 0] += inputData[b + 0] * weightVal;
			//		outputData[a + 1] += inputData[b + 1] * weightVal;
			//		outputData[a + 2] += inputData[b + 2] * weightVal;
			//		outputData[a + 3] += inputData[b + 3] * weightVal;
			//		outputData[a + 4] += inputData[b + 4] * weightVal;
			//		outputData[a + 5] += inputData[b + 5] * weightVal;
			//		outputData[a + 6] += inputData[b + 6] * weightVal;
			//		outputData[a + 7] += inputData[b + 7] * weightVal;
			//		outputData[a + 8] += inputData[b + 8] * weightVal;
			//		outputData[a + 9] += inputData[b + 9] * weightVal;

			//		i += vectorizeCount;
			//	}
			//}



			//int i = 0;
			//int j = 0;
			//repeat = height / vectorizeCount;
			//repeat *= repeat;
			//while (repeat-- > 0)
			//{
			//	// ver 1
			//	
			//	tempInputPos = inChIndex + rowIndex + i;
			//	tempOutputPos = outChIndex + rowIndex + i;

			//	outputData[tempOutputPos + 0] += inputData[tempInputPos + 0] * weightVal;
			//	outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
			//	outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
			//	outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
			//	outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
			//	outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
			//	outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
			//	outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
			//	outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
			//	outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

			//	outputData[tempOutputPos + 10] += inputData[tempInputPos + 10] * weightVal;
			//	outputData[tempOutputPos + 11] += inputData[tempInputPos + 11] * weightVal;
			//	outputData[tempOutputPos + 12] += inputData[tempInputPos + 12] * weightVal;
			//	outputData[tempOutputPos + 13] += inputData[tempInputPos + 13] * weightVal;
			//	outputData[tempOutputPos + 14] += inputData[tempInputPos + 14] * weightVal;
			//	outputData[tempOutputPos + 15] += inputData[tempInputPos + 15] * weightVal;
			//	outputData[tempOutputPos + 16] += inputData[tempInputPos + 16] * weightVal;
			//	outputData[tempOutputPos + 17] += inputData[tempInputPos + 17] * weightVal;
			//	outputData[tempOutputPos + 18] += inputData[tempInputPos + 18] * weightVal;
			//	outputData[tempOutputPos + 19] += inputData[tempInputPos + 19] * weightVal;

			//	outputData[tempOutputPos + 20] += inputData[tempInputPos + 20] * weightVal;
			//	outputData[tempOutputPos + 21] += inputData[tempInputPos + 21] * weightVal;
			//	outputData[tempOutputPos + 22] += inputData[tempInputPos + 22] * weightVal;
			//	outputData[tempOutputPos + 23] += inputData[tempInputPos + 23] * weightVal;
			//	outputData[tempOutputPos + 24] += inputData[tempInputPos + 24] * weightVal;
			//	outputData[tempOutputPos + 25] += inputData[tempInputPos + 25] * weightVal;
			//	outputData[tempOutputPos + 26] += inputData[tempInputPos + 26] * weightVal;
			//	outputData[tempOutputPos + 27] += inputData[tempInputPos + 27] * weightVal;
			//	outputData[tempOutputPos + 28] += inputData[tempInputPos + 28] * weightVal;
			//	outputData[tempOutputPos + 29] += inputData[tempInputPos + 29] * weightVal;

			//	outputData[tempOutputPos + 30] += inputData[tempInputPos + 30] * weightVal;
			//	outputData[tempOutputPos + 31] += inputData[tempInputPos + 31] * weightVal;
			//	outputData[tempOutputPos + 32] += inputData[tempInputPos + 32] * weightVal;
			//	outputData[tempOutputPos + 33] += inputData[tempInputPos + 33] * weightVal;
			//	outputData[tempOutputPos + 34] += inputData[tempInputPos + 34] * weightVal;
			//	outputData[tempOutputPos + 35] += inputData[tempInputPos + 35] * weightVal;
			//	outputData[tempOutputPos + 36] += inputData[tempInputPos + 36] * weightVal;
			//	outputData[tempOutputPos + 37] += inputData[tempInputPos + 37] * weightVal;
			//	outputData[tempOutputPos + 38] += inputData[tempInputPos + 38] * weightVal;
			//	outputData[tempOutputPos + 39] += inputData[tempInputPos + 39] * weightVal;

			//	outputData[tempOutputPos + 40] += inputData[tempInputPos + 40] * weightVal;
			//	outputData[tempOutputPos + 41] += inputData[tempInputPos + 41] * weightVal;
			//	outputData[tempOutputPos + 42] += inputData[tempInputPos + 42] * weightVal;
			//	outputData[tempOutputPos + 43] += inputData[tempInputPos + 43] * weightVal;
			//	outputData[tempOutputPos + 44] += inputData[tempInputPos + 44] * weightVal;
			//	outputData[tempOutputPos + 45] += inputData[tempInputPos + 45] * weightVal;
			//	outputData[tempOutputPos + 46] += inputData[tempInputPos + 46] * weightVal;
			//	outputData[tempOutputPos + 47] += inputData[tempInputPos + 47] * weightVal;
			//	outputData[tempOutputPos + 48] += inputData[tempInputPos + 48] * weightVal;
			//	outputData[tempOutputPos + 49] += inputData[tempInputPos + 49] * weightVal;

			//	outputData[tempOutputPos + 50] += inputData[tempInputPos + 50] * weightVal;
			//	outputData[tempOutputPos + 51] += inputData[tempInputPos + 51] * weightVal;
			//	outputData[tempOutputPos + 52] += inputData[tempInputPos + 52] * weightVal;
			//	outputData[tempOutputPos + 53] += inputData[tempInputPos + 53] * weightVal;
			//	outputData[tempOutputPos + 54] += inputData[tempInputPos + 54] * weightVal;
			//	outputData[tempOutputPos + 55] += inputData[tempInputPos + 55] * weightVal;
			//	outputData[tempOutputPos + 56] += inputData[tempInputPos + 56] * weightVal;
			//	outputData[tempOutputPos + 57] += inputData[tempInputPos + 57] * weightVal;
			//	outputData[tempOutputPos + 58] += inputData[tempInputPos + 58] * weightVal;
			//	outputData[tempOutputPos + 59] += inputData[tempInputPos + 59] * weightVal;

			//	outputData[tempOutputPos + 60] += inputData[tempInputPos + 60] * weightVal;
			//	outputData[tempOutputPos + 61] += inputData[tempInputPos + 61] * weightVal;
			//	outputData[tempOutputPos + 62] += inputData[tempInputPos + 62] * weightVal;
			//	outputData[tempOutputPos + 63] += inputData[tempInputPos + 63] * weightVal;
			//	outputData[tempOutputPos + 64] += inputData[tempInputPos + 64] * weightVal;
			//	outputData[tempOutputPos + 65] += inputData[tempInputPos + 65] * weightVal;
			//	outputData[tempOutputPos + 66] += inputData[tempInputPos + 66] * weightVal;
			//	outputData[tempOutputPos + 67] += inputData[tempInputPos + 67] * weightVal;
			//	outputData[tempOutputPos + 68] += inputData[tempInputPos + 68] * weightVal;
			//	outputData[tempOutputPos + 69] += inputData[tempInputPos + 69] * weightVal;

			//	outputData[tempOutputPos + 70] += inputData[tempInputPos + 70] * weightVal;
			//	outputData[tempOutputPos + 71] += inputData[tempInputPos + 71] * weightVal;
			//	outputData[tempOutputPos + 72] += inputData[tempInputPos + 72] * weightVal;
			//	outputData[tempOutputPos + 73] += inputData[tempInputPos + 73] * weightVal;
			//	outputData[tempOutputPos + 74] += inputData[tempInputPos + 74] * weightVal;
			//	outputData[tempOutputPos + 75] += inputData[tempInputPos + 75] * weightVal;
			//	outputData[tempOutputPos + 76] += inputData[tempInputPos + 76] * weightVal;
			//	outputData[tempOutputPos + 77] += inputData[tempInputPos + 77] * weightVal;
			//	outputData[tempOutputPos + 78] += inputData[tempInputPos + 78] * weightVal;
			//	outputData[tempOutputPos + 79] += inputData[tempInputPos + 79] * weightVal;

			//	outputData[tempOutputPos + 80] += inputData[tempInputPos + 80] * weightVal;
			//	outputData[tempOutputPos + 81] += inputData[tempInputPos + 81] * weightVal;
			//	outputData[tempOutputPos + 82] += inputData[tempInputPos + 82] * weightVal;
			//	outputData[tempOutputPos + 83] += inputData[tempInputPos + 83] * weightVal;
			//	outputData[tempOutputPos + 84] += inputData[tempInputPos + 84] * weightVal;
			//	outputData[tempOutputPos + 85] += inputData[tempInputPos + 85] * weightVal;
			//	outputData[tempOutputPos + 86] += inputData[tempInputPos + 86] * weightVal;
			//	outputData[tempOutputPos + 87] += inputData[tempInputPos + 87] * weightVal;
			//	outputData[tempOutputPos + 88] += inputData[tempInputPos + 88] * weightVal;
			//	outputData[tempOutputPos + 89] += inputData[tempInputPos + 89] * weightVal;

			//	outputData[tempOutputPos + 90] += inputData[tempInputPos + 90] * weightVal;
			//	outputData[tempOutputPos + 91] += inputData[tempInputPos + 91] * weightVal;
			//	outputData[tempOutputPos + 92] += inputData[tempInputPos + 92] * weightVal;
			//	outputData[tempOutputPos + 93] += inputData[tempInputPos + 93] * weightVal;
			//	outputData[tempOutputPos + 94] += inputData[tempInputPos + 94] * weightVal;
			//	outputData[tempOutputPos + 95] += inputData[tempInputPos + 95] * weightVal;
			//	outputData[tempOutputPos + 96] += inputData[tempInputPos + 96] * weightVal;
			//	outputData[tempOutputPos + 97] += inputData[tempInputPos + 97] * weightVal;
			//	outputData[tempOutputPos + 98] += inputData[tempInputPos + 98] * weightVal;
			//	outputData[tempOutputPos + 99] += inputData[tempInputPos + 99] * weightVal;



			//	// ver 2
			//	//rowIndex = j * width;
			//	//tempOutputData = outputData + inChIndex + rowIndex + i;
			//	//tempInputData = inputData + inChIndex + rowIndex + i;

			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	//tempInputData += width - vectorizeCount;
			//	//tempOutputData += width - vectorizeCount;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;
			//	//*tempOutputData++ += *tempInputData++ * weightVal;

			//	i += vectorizeCount;
			//	if (i >= width)
			//	{
			//		i = 0;
			//		j += vectorizeCount;
			//		rowIndex = j * width;
			//	}
			//}
			++weight;
		}

		for (int r = 0; r < height; ++r)
		{
			for (int c = 0; c < width; ++c)
			{
				T val = outputData[outCh * area + r * width + c] + bias[outCh];
				outputData[outCh * area + r * width + c] = (val < 0) ? 0 : val;
			}
		}


		//T* v = outputData + outCh * area - 1;
		//T biasVal = bias[outCh];
		//for (int r = 0; r < area; ++r)
		//{
		//	T val = *++v + biasVal;
		//	*v = (val < 0) ? 0 : val;
		//}
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}






// yet no need to implement
template <typename T>
std::chrono::microseconds Convolution2D_Pointwise_k1_s2(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}
