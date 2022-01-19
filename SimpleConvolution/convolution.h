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


// vectorized version

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
//#pragma region padding
//
//	// make padding tensor
//	float* padInput = new float[inChannel * padHeight * padWidth];
//	for (int i = 0; i < inChannel * padHeight * padWidth; ++i)
//	{
//		padInput[i] = 0;
//	}
//
//	ZeroPadding(inputData, padInput, height, width, inChannel, padding);
//
//#pragma endregion
//
//	int kernelSize = inChannel * 9;
//	int vectorizeCount = 10 * stride;
//	int repeat = padWidth / vectorizeCount;
//	repeat *= repeat;
//
//	int tempTopInputPos;
//	int tempMidInputPos;
//	int tempBotInputPos;
//	int tempOutputPos;
//
//	//T val_1, val_2, val_3, val_4, val_5, val_6;
//
//	for (int outCh = 0; outCh < outChannel; ++outCh)
//	{
//		int outChIndex = outCh * outputArea;
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
//			for (int row = 0; row < padHeight; row += stride)
//			{
//				int topRow = inChIndex + row * padWidth;
//				int midRow = inChIndex + (row+1) * padWidth;
//				int botRow = inChIndex + (row+2) * padWidth;
//				int outRow = outChIndex + row / stride * outputWidth;
//				for (int col = 0; col < padWidth; col += stride)
//				{
//					int topInputPos = topRow + col;
//					int midInputPos = midRow + col;
//					int botInputPos = botRow + col;
//					int outputPos = outRow + col/stride;
//					outputData[outputPos] += padInput[topInputPos] * weightVal_1 + padInput[topInputPos + 1] * weightVal_2 + padInput[topInputPos + 2] * weightVal_3 +
//						padInput[midInputPos] * weightVal_4 + padInput[midInputPos + 1] * weightVal_5 + padInput[midInputPos + 2] * weightVal_6 +
//						padInput[botInputPos] * weightVal_7 + padInput[botInputPos + 1] * weightVal_8 + padInput[botInputPos + 2] * weightVal_9;
//
//					//outputData[outputPos] += padInput[topInputPos] * weightVal_1;
//					//outputData[outputPos] += padInput[topInputPos +1] * weightVal_2;
//					//outputData[outputPos] += padInput[topInputPos +2] * weightVal_3;
//					//outputData[outputPos] += padInput[midInputPos] * weightVal_4;
//					//outputData[outputPos] += padInput[midInputPos +1] * weightVal_5;
//					//outputData[outputPos] += padInput[midInputPos +2] * weightVal_6;
//					//outputData[outputPos] += padInput[botInputPos] * weightVal_7;
//					//outputData[outputPos] += padInput[botInputPos +1] * weightVal_8;
//					//outputData[outputPos] += padInput[botInputPos +2] * weightVal_9;
//
//				}
//			}
//
//
//
////			#pragma region vectorize
////
////			int i = 0;
////			int j = 0;
////			repeat = padWidth / vectorizeCount;
////			repeat *= repeat;
////			while (repeat-- > 0)
////			{
////				tempTopInputPos = inChIndex + j * padWidth + i;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = outChIndex + j / stride * outputWidth + i / stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
////
////
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
////
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
////
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
////
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
////
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
////
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
////
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
////
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos = tempTopInputPos + padWidth - (vectorizeCount - 1) * stride;
////				tempMidInputPos = tempTopInputPos + padWidth;
////				tempBotInputPos = tempMidInputPos + padWidth;
////				tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
////
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////				tempTopInputPos += stride;
////				tempMidInputPos += stride;
////				tempBotInputPos += stride;
////
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
////				outputData[tempOutputPos] += padInput[tempTopInputPos + 2] * weightVal_3;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
////				outputData[tempOutputPos] += padInput[tempMidInputPos + 2] * weightVal_6;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
////				outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
////				outputData[tempOutputPos++] += padInput[tempBotInputPos + 2] * weightVal_9;
////
////				i += vectorizeCount;
////				if (i >= width)
////				{
////					i = 0;
////					j += vectorizeCount;
////				}
////			}
////
////#pragma endregion
////
//
//
//		}
//		for (int r = 0; r < outputHeight; ++r)
//		{
//			for (int c = 0; c < outputWidth; ++c)
//			{
//				T val = outputData[outCh * outputArea + r * outputWidth + c] + bias[outCh];
//				//outputData[outCh * outputArea + r * outputWidth + c] = val;
//				outputData[outCh * outputArea + r * outputWidth + c] = (val < 0) ? 0 : val;
//			}
//		}
//	}
//
//	delete[]padInput;
//
//	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
//	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//}



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

	//for (int i = 0; i < padHeight; ++i)
	//{
	//	for (int j = 0; j < padWidth; ++j)
	//	{
	//		if (j < 5 || j > padWidth - 6)
	//			std::cout << padInput[i * padWidth + j] << ", ";
	//	}
	//	std::cout << std::endl;
	//}

	#pragma endregion

	int kernelSize = inChannel * 9;

	T val_1, val_2, val_3, val_4, val_5, val_6;

	T* tempInputData1 = padInput;
	T* tempInputData2 = padInput + padWidth;
	T* tempInputData3 = padInput + padWidth + padWidth;
	T* tempOutputData = outputData;

	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		int outKernel = outCh * kernelSize;
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			int kernelIndex = outKernel + inCh * 9;

			T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
			T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
			T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

			for (int row = 0; row < height; row += stride)
			{
				for (int col = 0; col < width; col += stride, tempInputData1 += stride, tempInputData2 += stride, tempInputData3 += stride, ++outputData)
				{
					T val = *outputData;

					T val1 = *(tempInputData1);
					T val2 = *(tempInputData1 + 1);
					T val3 = *(tempInputData1 + 2);

					T val4 = *(tempInputData2);
					T val5 = *(tempInputData2 + 1);
					T val6 = *(tempInputData2 + 2);

					T val7 = *(tempInputData3);
					T val8 = *(tempInputData3 + 1);
					T val9 = *(tempInputData3 + 2);

					val = val + val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
						val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
						val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9;
					*outputData = val;

					//tempInputData1 += stride;
					//tempInputData2 += stride;
					//tempInputData3 += stride;
					//++outputData;
				}
				tempInputData1 += padWidth + 1;
				tempInputData2 += padWidth + 1;
				tempInputData3 += padWidth + 1;
			}
			tempInputData1 += padWidth + 1;
			tempInputData2 += padWidth + 1;
			tempInputData3 += padWidth + 1;

			outputData = tempOutputData;
		}
		for (int i = 0; i < area; ++i)
		{
			T val = *outputData + *bias;
			val = (val < 0) ? 0 : val;
			*outputData = val;

			++outputData;
		}
		++bias;
		tempOutputData = outputData;

		tempInputData1 = padInput;
		tempInputData2 = padInput + padWidth;
		tempInputData3 = padInput + padWidth + padWidth;

	}
	delete[]padInput;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}








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


//// vectorized version

//
//template <typename T>
//std::chrono::microseconds Convolution2D_Depthwise_k3_s1(T* inputData, T* outputData, T* weight, T* bias,
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
//	int outputHeight = (padHeight - kernel) + 1;
//	int outputWidth = (padWidth - kernel) + 1;
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
//	int vectorizeCount = 10;
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
//	for (int inCh = 0; inCh < inChannel; ++inCh)
//	{
//		int inChIndex = inCh * padArea;
//		int kernelIndex = inCh * 9;
//
//		T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
//		T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
//		T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];
//
//		int i = 0;
//		int j = 0;
//		repeat = padWidth / vectorizeCount;
//		repeat *= repeat;
//		while (repeat-- > 0)
//		{
//			tempTopInputPos = inChIndex + j * padWidth + i;
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = inChIndex + j / stride * outputWidth + i / stride;
//
//#pragma region vectorized
//			// row 1
//			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
//			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
//			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
//			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//
//			// row 2
//			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
//			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
//			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
//			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 3
//			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
//			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
//			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
//			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 4
//			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
//			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
//			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
//			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 5
//			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
//			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
//			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
//			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 6
//			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
//			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
//			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
//			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			tempTopInputPos = tempTopInputPos + padWidth - vectorizeCount - 1;
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 7
//			outputData[tempOutputPos] += (padInput[tempTopInputPos + 0] * weightVal_1) + (padInput[tempMidInputPos + 0] * weightVal_4) + (padInput[tempBotInputPos + 0] * weightVal_7);
//			val_1 = padInput[tempTopInputPos + 1] * weightVal_2;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_3;
//			val_3 = padInput[tempMidInputPos + 1] * weightVal_5;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_6;
//			val_5 = padInput[tempBotInputPos + 1] * weightVal_8;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//			++tempTopInputPos;
//			++tempMidInputPos;
//			++tempBotInputPos;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3 + val_4 + val_5 + val_6;
//			val_2 = padInput[tempTopInputPos + 2] * weightVal_1;
//			val_4 = padInput[tempMidInputPos + 2] * weightVal_4;
//			val_6 = padInput[tempBotInputPos + 2] * weightVal_7;
//			outputData[tempOutputPos++] += val_2 + val_4 + val_6;
//#pragma endregion
//
//			i += vectorizeCount;
//			if (i >= width)
//			{
//				i = 0;
//				j += vectorizeCount;
//			}
//		}
//
//		for (int r = 0; r < outputHeight; ++r)
//		{
//			for (int c = 0; c < outputWidth; ++c)
//			{
//				T val = outputData[inCh * outputArea + r * outputWidth + c] + bias[inCh];
//				outputData[inCh * outputArea + r * outputWidth + c] = (val < 0) ? 0 : val;
//			}
//		}
//	}
//	delete[]padInput;
//
//	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
//	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//}

//template <typename T>
//std::chrono::microseconds Convolution2D_Depthwise_k3_s2(T* inputData, T* outputData, T* weight, T* bias,
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
//	int vectorizeCount = 10 * stride;
//	int repeat = padWidth / vectorizeCount;
//	repeat *= repeat;
//
//	int tempTopInputPos;
//	int tempMidInputPos;
//	int tempBotInputPos;
//	int tempOutputPos;
//
//	T val_1;
//	T val_2;
//	T val_3;
//
//	int count = 0;
//
//	for (int inCh = 0; inCh < inChannel; ++inCh)
//	{
//		int inChIndex = inCh * padArea;
//		int outChIndex = inCh * outputArea;
//		int kernelIndex = inCh * 9;
//
//		T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
//		T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
//		T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];
//
//		int i = 0;
//		int j = 0;
//		repeat = padWidth / vectorizeCount;
//		repeat *= repeat;
//		while (repeat-- > 0)
//		{
//			//count++;
//			tempTopInputPos = inChIndex + j * padWidth + i;
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = outChIndex + j / stride * outputWidth + i / stride;
//
//#pragma region vectorized
//
//			// row 1
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 2
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 3
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 4
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 5
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 6
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 7
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 8
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 9
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos = tempTopInputPos + ((padWidth - vectorizeCount - 1) * stride);
//			tempMidInputPos = tempTopInputPos + padWidth;
//			tempBotInputPos = tempMidInputPos + padWidth;
//			tempOutputPos = tempOutputPos - vectorizeCount + outputWidth;
//
//			// row 10
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 0] * weightVal_1;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 0] * weightVal_4;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 0] * weightVal_7;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//			tempTopInputPos += stride;
//			tempMidInputPos += stride;
//			tempBotInputPos += stride;
//
//			outputData[tempOutputPos] += val_1 + val_2 + val_3;
//			outputData[tempOutputPos] += padInput[tempTopInputPos + 1] * weightVal_2;
//			val_1 = padInput[tempTopInputPos + 2] * weightVal_3;
//			outputData[tempOutputPos] += val_1;
//			outputData[tempOutputPos] += padInput[tempMidInputPos + 1] * weightVal_5;
//			val_2 = padInput[tempMidInputPos + 2] * weightVal_6;
//			outputData[tempOutputPos] += val_2;
//			outputData[tempOutputPos] += padInput[tempBotInputPos + 1] * weightVal_8;
//			val_3 = padInput[tempBotInputPos + 2] * weightVal_9;
//			outputData[tempOutputPos++] += val_3;
//
//#pragma endregion
//
//			i += vectorizeCount;
//			if (i >= width)
//			{
//				i = 0;
//				j += vectorizeCount;
//			}
//		}
//
//		for (int r = 0; r < outputHeight; ++r)
//		{
//			for (int c = 0; c < outputWidth; ++c)
//			{
//				T val = outputData[inCh * outputArea + r * outputWidth + c] + bias[inCh];
//				outputData[inCh * outputArea + r * outputWidth + c] = (val < 0) ? 0 : val;
//			}
//		}
//	}
//	delete[]padInput;
//	//std::cout << "depthwise : " << count << std::endl;
//
//	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
//	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//}



//// plain

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


	T val_1, val_2, val_3, val_4, val_5, val_6;

	T* tempInputData1 = padInput;
	T* tempInputData2 = padInput + padWidth;
	T* tempInputData3 = padInput + padWidth + padWidth;
	T* tempOutputData = outputData;

	for (int inCh = 0; inCh < inChannel; ++inCh)
	{
		int kernelIndex = inCh * 9;

		T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
		T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
		T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

		for (int row = 0; row < height; ++row)
		{
			for (int col = 0; col < width; ++col)
			{
				T val = *outputData;

				T val1 = *(tempInputData1);
				T val2 = *(tempInputData1 + 1);
				T val3 = *(tempInputData1 + 2);

				T val4 = *(tempInputData2);
				T val5 = *(tempInputData2 + 1);
				T val6 = *(tempInputData2 + 2);

				T val7 = *(tempInputData3);
				T val8 = *(tempInputData3 + 1);
				T val9 = *(tempInputData3 + 2);

				val = val + val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
					val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
					val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9;

				++tempInputData1;
				++tempInputData2;
				++tempInputData3;
				++outputData;
			}
			tempInputData1 += 2;
			tempInputData2 += 2;
			tempInputData3 += 2;
		}
		tempInputData1 += 2;
		tempInputData2 += 2;
		tempInputData3 += 2;
		
		outputData = tempOutputData;
		for (int i = 0; i < area; ++i)
		{
			T val = *outputData + *bias;
			val = (val < 0) ? 0 : val;
			*outputData = val;

			++outputData;
		}
		++bias;
		tempOutputData = outputData;

		//for (int r = 0; r < outputHeight; ++r)
		//{
		//	for (int c = 0; c < outputWidth; ++c)
		//	{
		//		T val = outputData[inCh * outputArea + r * outputWidth + c] + bias[inCh];
		//		outputData[inCh * outputArea + r * outputWidth + c] = (val < 0) ? 0 : val;
		//	}
		//}
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

	T val_1, val_2, val_3, val_4, val_5, val_6;

	T* tempInputData1 = padInput;
	T* tempInputData2 = padInput + padWidth;
	T* tempInputData3 = padInput + padWidth + padWidth;
	T* tempOutputData = outputData;

	for (int inCh = 0; inCh < inChannel; ++inCh)
	{
		int kernelIndex = inCh * 9;

		T weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
		T weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
		T weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

		for (int row = 0; row < height; row+=stride)
		{
			for (int col = 0; col < width; col+=stride)
			{
				T val = *outputData;

				T val1 = *(tempInputData1);
				T val2 = *(tempInputData1 + 1);
				T val3 = *(tempInputData1 + 2);

				T val4 = *(tempInputData2);
				T val5 = *(tempInputData2 + 1);
				T val6 = *(tempInputData2 + 2);

				T val7 = *(tempInputData3);
				T val8 = *(tempInputData3 + 1);
				T val9 = *(tempInputData3 + 2);

				val = val + val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
					val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
					val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9;
				*outputData = val;
				tempInputData1 += stride;
				tempInputData2 += stride;
				tempInputData3 += stride;
				++outputData;
			}
			tempInputData1 += padWidth + 1;
			tempInputData2 += padWidth + 1;
			tempInputData3 += padWidth + 1;
		}
		tempInputData1 += padWidth + 1;
		tempInputData2 += padWidth + 1;
		tempInputData3 += padWidth + 1;

		outputData = tempOutputData;
		for (int i = 0; i < area; ++i)
		{
			T val = *outputData + *bias;
			val = (val < 0) ? 0 : val;
			*outputData = val;

			++outputData;
		}
		++bias;
		tempOutputData = outputData;
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



template <typename T>
std::chrono::microseconds Convolution2D_Pointwise_k1_s1(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int area = height * width;
	T* tempOutputData = outputData;
	T* tempInputData = inputData;

	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		for (int i = 0; i < area; ++i)
		{
			T val = 0.0f;
			for (int inCh = 0; inCh < inChannel; ++inCh)
			{
				val += inputData[inCh * area + i] * weight[inCh];
			}
			val = val + *bias;
			val = (val < 0) ? 0 : val;
			*outputData = val;

			++outputData;
		}
		++bias;
	
		



		//int outPos = outCh * area;
		//for (int inCh = 0; inCh < inChannel; ++inCh)
		//{
		//	T weightVal = *weight;
		//	outputData = tempOutputData;
		//	int inPos = inCh * area;
		//	
		//	
		//	for (int i = 0; i < area; ++i)
		//	{
		//		T o = *outputData;
		//		//T o = outputData[outPos + i];
		//		//T p = *inputData++;
		//		//o = o + p * weightVal;
		//		o = o + inputData[inPos + i] * weightVal;
		//		//outputData[outPos + i] = o;
		//		*outputData = o;
		//		++outputData;
		//	}
		//}
		//outputData = tempOutputData;
		//inputData = tempInputData;


		//for (int i = 0; i < area; ++i)
		//{
		//	T val = *outputData + *bias;
		//	val = (val < 0) ? 0 : val;
		//	*outputData = val;

		//	++outputData;
		//}
		//tempOutputData = outputData;

		//++bias;
		//++weight;
	}



	//for (int outCh = 0; outCh < outChannel; ++outCh, ++bias, ++weight, tempOutputData+=area)
	//{
	//	for (int inCh = 0; inCh < inChannel; ++inCh)
	//	{
	//		T weightVal = *weight;
	//		outputData = tempOutputData;
	//		for (int i = 0; i < area; ++i, ++outputData, ++inputData)
	//		{
	//			T o = *outputData;
	//			T p = *inputData;
	//			T f = p * weightVal;
	//			o = o + *inputData * weightVal;
	//			*outputData = o;

	//			//++outputData;
	//			//++inputData;
	//		}
	//	}
	//	outputData = tempOutputData;
	//	inputData = tempInputData;

	//	for (int i = 0; i < area; ++i)
	//	{
	//		T val = *outputData + *bias;
	//		val = (val < 0) ? 0 : val;
	//		*outputData = val;
	//		
	//		++outputData;
	//	}
	//	//++bias;
	//	//tempOutputData += area;
	//	//++weight;
	//}

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
