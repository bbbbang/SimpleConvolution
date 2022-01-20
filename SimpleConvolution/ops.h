#pragma once
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>

template <typename T>
std::chrono::microseconds ZeroPadding(T* inputData, T* outputData, int height, int width, int channel, int padding);

template <typename T>
std::chrono::microseconds Add(T* inputData, T* outputData, int height, int width, int channel);

template <typename T>
std::chrono::microseconds Relu(T* inputData, int height, int width, int channel);

template <typename T>
std::chrono::microseconds Concat(T* inputData, T* outputData, int height, int width, int channel);


// vectorize

//template <typename T>
//std::chrono::microseconds ZeroPadding(T* inputData, T* outputData, int height, int width, int channel, int padding)
//{
//	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
//
//	int vectorizeCount = 10;
//	int repeat = height / vectorizeCount;
//	repeat *= repeat;
//
//	int inputIndex;
//	int outputIndex;
//
//	int inputArea = height * width;
//	int outputWidth = (height + padding * 2);
//	int outputArea = outputWidth * outputWidth;
//
//	for (int ch = 0; ch < channel; ++ch)
//	{
//#pragma region vectorize
//
//		int i = 0;
//		int j = 0;
//		repeat = height / vectorizeCount;
//		repeat *= repeat;
//		while (repeat-- > 0)
//		{
//			outputIndex = ch * outputArea + (j + 1) * outputWidth + (i + 1);
//			inputIndex = ch * inputArea + j * width + i;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//			outputIndex += outputWidth;
//			inputIndex += width;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//			outputIndex += outputWidth;
//			inputIndex += width;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//			outputIndex += outputWidth;
//			inputIndex += width;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//			outputIndex += outputWidth;
//			inputIndex += width;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//			outputIndex += outputWidth;
//			inputIndex += width;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//			outputIndex += outputWidth;
//			inputIndex += width;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//			outputIndex += outputWidth;
//			inputIndex += width;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//			outputIndex += outputWidth;
//			inputIndex += width;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//			outputIndex += outputWidth;
//			inputIndex += width;
//
//			outputData[outputIndex + 0] = inputData[inputIndex + 0];
//			outputData[outputIndex + 1] = inputData[inputIndex + 1];
//			outputData[outputIndex + 2] = inputData[inputIndex + 2];
//			outputData[outputIndex + 3] = inputData[inputIndex + 3];
//			outputData[outputIndex + 4] = inputData[inputIndex + 4];
//			outputData[outputIndex + 5] = inputData[inputIndex + 5];
//			outputData[outputIndex + 6] = inputData[inputIndex + 6];
//			outputData[outputIndex + 7] = inputData[inputIndex + 7];
//			outputData[outputIndex + 8] = inputData[inputIndex + 8];
//			outputData[outputIndex + 9] = inputData[inputIndex + 9];
//
//			i += vectorizeCount;
//			if (i >= height)
//			{
//				i = 0;
//				j += vectorizeCount;
//			}
//		}
//#pragma endregion
//	}
//
//	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
//	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//}



template <typename T>
std::chrono::microseconds ZeroPadding(T* inputData, T* outputData, int height, int width, int channel, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int inputIndex;
	int outputIndex;

	int inputArea = height * width;
	int outputWidth = (height + padding * 2);
	int outputArea = outputWidth * outputWidth;

	for (int ch = 0; ch < channel; ++ch)
	{
		outputData += outputWidth;
		for (int row = 0; row < height; ++row)
		{
			++outputData;
			for (int col = 0; col < width; ++col)
			{
				T val = *inputData;
				*outputData = val;

				//outputData[ch * outputArea + (row+1) * outputWidth + (col+1)] = inputData[ch * inputArea + row * width + col];

				++inputData;
				++outputData;
			}
			++outputData;
		}
		outputData += outputWidth;
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}




template <typename T>
std::chrono::microseconds Add(T* inputData, T* outputData, int height, int width, int channel)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	for (int i = 0; i < height * width * channel; ++i)
	{
		//*outputData++ = *inputData++ + *outputData;

		T out = *outputData;
		T in = *inputData;

		out += in;
		*outputData = out;

		++outputData;
		++inputData;
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

template <typename T>
std::chrono::microseconds Relu(T* inputData, int height, int width, int channel)
{
	for (int i = 0; i < height * width * channel; ++i)
	{
		*inputData++ = (*inputData < 0) ? 0 : *inputData;
	}
}

template <typename T>
std::chrono::microseconds Concat(T* inputData, T* outputData, int height, int width, int channel)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int size = height * width * channel;
	for (int i = size; i < size * 2; ++i)
	{
		inputData[i] = outputData[i - size];
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}


template <typename T>
std::chrono::microseconds ZeroConcat(T* inputData, T* outputData, int height, int width, int channel);
template <typename T>
std::chrono::microseconds ZeroConcat(T* inputData, T* outputData, int height, int width, int channel)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int size = height * width * channel;
	for (int i = size; i < size * 2; ++i)
	{
		inputData[i] = 0;
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

template <typename T>
std::chrono::microseconds MaxPool(T* inputData, T* outputData, int height, int width, int channel, int kernel, int stride, int padding);

template <typename T>
std::chrono::microseconds MaxPool(T* inputData, T* outputData, int height, int width, int channel, int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int padHeight = height + padding * 2;
	int padWidth = width + padding * 2;
	int padArea = padWidth * padHeight;

	int outputHeight = (padHeight - kernel) / stride + 1;
	int outputWidth = (padWidth - kernel) / stride + 1;
	int outputArea = outputHeight * outputWidth;

	for (int ch = 0; ch < channel; ++ch)
	{
		for (int row = 0; row < height; row+=stride)
		{
			for (int col = 0; col < width; col += stride)
			{
				T val1 = *inputData;
				T val2 = *(inputData + 1);
				T val3 = *(inputData + width);
				T val4 = *(inputData + width + 1);

				T max = std::max({ val1, val2, val3, val4 });
				*outputData = max;

				inputData+= stride;
				++outputData;
			}
		}
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}


template <typename T>
std::chrono::microseconds Resize(T* inputData, T* outputData, int height, int width, int channel, float scale);

template <typename T>
std::chrono::microseconds Resize(T* inputData, T* outputData, int height, int width, int channel, float scale)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
	// nearest neighbor

	int outputHeight = height * scale;
	int outputWidth = width * scale;
	int outputArea = outputHeight * outputWidth;

	int area = height * width;

	for (int i = 0; i < channel * outputArea; ++i)
	{
		outputData[i] = 0;
	}

	for (int ch = 0; ch < channel; ++ch)
	{
		for (int row = 0; row < height; ++row)
		{
			for (int col = 0; col < width; ++col)
			{
				T val = *inputData;

				*outputData = val;
				*(outputData + 1) = val;
				*(outputData + outputWidth) = val;
				*(outputData + outputWidth + 1) = val;

				++inputData;
				outputData+=2;
			}
		}

	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}
