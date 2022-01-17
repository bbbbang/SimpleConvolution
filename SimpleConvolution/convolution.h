#pragma once
#include "ops.h"

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

	// padding
	float* padInput = new float[inChannel * padHeight * padWidth];

	for (int i = 0; i < inChannel * padHeight * padWidth; ++i)
	{
		padInput[i] = 0;
	}

	ZeroPadding(inputData, padInput, height, width, inChannel, padding);

	int kernelSize = inChannel * 9;
	//int vectorizeCount = 10 * stride;
	int vectorizeCount = 10 * stride;
	int repeat = padWidth / vectorizeCount;
	repeat *= repeat;

	int tempTopInputPos;
	int tempMidInputPos;
	int tempBotInputPos;
	int tempOutputPos;

	T val_1, val_2, val_3, val_4, val_5, val_6;

	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		int outChIndex = outCh * padArea;
		int kernelArea = outCh * kernelSize;
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			int inChIndex = inCh * padArea;
			int kernelIndex = kernelArea + inCh * 9;

			T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
			T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
			T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

			int i = 0;
			int j = 0;
			while (repeat-- > 0)
			{
				tempTopInputPos = inChIndex + j * padWidth + i;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = outChIndex + j/stride * outputWidth + i/stride;

				#pragma region vectorized
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
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;



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
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


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
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


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
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


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
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


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
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


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
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


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
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


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
				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
				tempMidInputPos = tempTopInputPos + padWidth;
				tempBotInputPos = tempMidInputPos + padWidth;
				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;


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
				if (i > padWidth)
				{
					i = 0;
					j += vectorizeCount;
				}
			}
		}
		for (int r = 0; r < outputHeight; ++r)
		{
			for (int c = 0; c < outputWidth; ++c)
			{
				T val = outputData[outCh * outputArea + r * outputWidth + c] + bias[outCh];
				outputData[outCh * outputArea + r * outputWidth + c] = val;
				//outputData[outCh * outputArea + r * outputWidth + c] = (val < 0) ? 0 : val;
			}
		}
	}

	delete[]padInput;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}


//
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
//	//int vectorizeCount = 10 * stride;
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
		while (repeat-- > 0)
		{
			tempTopInputPos = inChIndex + j * padWidth + i;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = inChIndex + j/stride * outputWidth + i/stride;

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
			if (i > padWidth)
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
	for (int inCh = 0; inCh < inChannel; ++inCh)
	{
		int inChIndex = inCh * padArea;
		int kernelIndex = inCh * 9;

		T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
		T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
		T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

		int i = 0;
		int j = 0;
		while (repeat-- > 0)
		{
			tempTopInputPos = inChIndex + j * padWidth + i;
			tempMidInputPos = tempTopInputPos + padWidth;
			tempBotInputPos = tempMidInputPos + padWidth;
			tempOutputPos = inChIndex + j/stride * outputWidth + i/stride;

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
			if (i > padWidth)
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




template <typename T>
std::chrono::microseconds Convolution2D_Pointwise_k1_s1(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int area = height * width;

	int rowIndex;
	int tempInputPos;
	int tempOutputPos;

	if (height % 10 == 0)
	{
		int vectorizeCount = 10;
		int repeat = height / vectorizeCount;
		repeat *= repeat;

		for (int outCh = 0; outCh < outChannel; ++outCh)
		{
			int outChIndex = outCh * area;
			for (int inCh = 0; inCh < inChannel; ++inCh)
			{
				int inChIndex = inCh * area;
				T weightVal = *weight;

				int i = 0;
				int j = 0;
				while (repeat-- > 0)
				{
					rowIndex = j * width;
					tempInputPos = inChIndex + rowIndex + i;
					tempOutputPos = outChIndex + rowIndex + i;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					rowIndex += width;
					tempInputPos += width;
					tempOutputPos += width;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					rowIndex += width;
					tempInputPos += width;
					tempOutputPos += width;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					rowIndex += width;
					tempInputPos += width;
					tempOutputPos += width;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					rowIndex += width;
					tempInputPos += width;
					tempOutputPos += width;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					rowIndex += width;
					tempInputPos += width;
					tempOutputPos += width;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					rowIndex += width;
					tempInputPos += width;
					tempOutputPos += width;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					rowIndex += width;
					tempInputPos += width;
					tempOutputPos += width;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					rowIndex += width;
					tempInputPos += width;
					tempOutputPos += width;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					rowIndex += width;
					tempInputPos += width;
					tempOutputPos += width;
					outputData[tempOutputPos] += inputData[tempInputPos] * weightVal;
					outputData[tempOutputPos + 1] += inputData[tempInputPos + 1] * weightVal;
					outputData[tempOutputPos + 2] += inputData[tempInputPos + 2] * weightVal;
					outputData[tempOutputPos + 3] += inputData[tempInputPos + 3] * weightVal;
					outputData[tempOutputPos + 4] += inputData[tempInputPos + 4] * weightVal;
					outputData[tempOutputPos + 5] += inputData[tempInputPos + 5] * weightVal;
					outputData[tempOutputPos + 6] += inputData[tempInputPos + 6] * weightVal;
					outputData[tempOutputPos + 7] += inputData[tempInputPos + 7] * weightVal;
					outputData[tempOutputPos + 8] += inputData[tempInputPos + 8] * weightVal;
					outputData[tempOutputPos + 9] += inputData[tempInputPos + 9] * weightVal;

					i += vectorizeCount;
					if (i > width)
					{
						i = 0;
						j += vectorizeCount;
					}
				}
				++weight;
			}
			T* v = outputData + outCh * area - 1;
			T biasVal = bias[outCh];
			for (int r = 0; r < area; ++r)
			{
				T val = *++v + biasVal;
				*v = (val < 0) ? 0 : val;
			}
		}
	}
	else
	{
		for (int outCh = 0; outCh < outChannel; ++outCh)
		{
			int outChIndex = outCh * area;
			for (int inCh = 0; inCh < inChannel; ++inCh)
			{
				int inChIndex = inCh * area;
				T weightVal = *weight;
				for (int row = 0; row < height; ++row)
				{
					int rowIndex = row * width;
					for (int col = 0; col < width; ++col)
					{
						int outputPos = outChIndex + rowIndex + col;
						outputData[outputPos] += inputData[inChIndex + rowIndex + col] * weightVal;
					}
				}
				++weight;
			}
			T* v = outputData + outCh * area - 1;
			T biasVal = bias[outCh];
			for (int r = 0; r < area; ++r)
			{
				T val = *++v + biasVal;
				*v = (val < 0) ? 0 : val;
			}
		}
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
