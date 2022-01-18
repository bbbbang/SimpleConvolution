#pragma once
#include <chrono>

template <typename T>
std::chrono::microseconds ZeroPadding(T* inputData, T* outputData, int height, int width, int channel, int padding);

template <typename T>
std::chrono::microseconds Add(T* inputData, T* outputData, int height, int width, int channel);

template <typename T>
std::chrono::microseconds Relu(T* inputData, int height, int width, int channel);

template <typename T>
std::chrono::microseconds Concat(T* inputData, T* outputData, int height, int width, int channel);

template <typename T>
std::chrono::microseconds ZeroPadding(T* inputData, T* outputData, int height, int width, int channel, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int vectorizeCount = 10;
	int repeat = height / vectorizeCount;
	repeat *= repeat;

	int inputIndex;
	int outputIndex;

	int inputArea = height * width;
	int outputWidth = (height + padding * 2);
	int outputArea = outputWidth * outputWidth;

	for (int ch = 0; ch < channel; ++ch)
	{
		#pragma region vectorize

		int i = 0;
		int j = 0;
		repeat = height / vectorizeCount;
		repeat *= repeat;
		while (repeat-- > 0)
		{
			outputIndex = ch * outputArea + (j + 1) * outputWidth + (i + 1);
			inputIndex = ch * inputArea + j * width + i;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];
			outputIndex += outputWidth;
			inputIndex += width;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];
			outputIndex += outputWidth;
			inputIndex += width;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];
			outputIndex += outputWidth;
			inputIndex += width;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];
			outputIndex += outputWidth;
			inputIndex += width;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];
			outputIndex += outputWidth;
			inputIndex += width;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];
			outputIndex += outputWidth;
			inputIndex += width;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];
			outputIndex += outputWidth;
			inputIndex += width;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];
			outputIndex += outputWidth;
			inputIndex += width;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];
			outputIndex += outputWidth;
			inputIndex += width;

			outputData[outputIndex + 0] = inputData[inputIndex + 0];
			outputData[outputIndex + 1] = inputData[inputIndex + 1];
			outputData[outputIndex + 2] = inputData[inputIndex + 2];
			outputData[outputIndex + 3] = inputData[inputIndex + 3];
			outputData[outputIndex + 4] = inputData[inputIndex + 4];
			outputData[outputIndex + 5] = inputData[inputIndex + 5];
			outputData[outputIndex + 6] = inputData[inputIndex + 6];
			outputData[outputIndex + 7] = inputData[inputIndex + 7];
			outputData[outputIndex + 8] = inputData[inputIndex + 8];
			outputData[outputIndex + 9] = inputData[inputIndex + 9];

			i += vectorizeCount;
			if (i >= height)
			{
				i = 0;
				j += vectorizeCount;
			}
		}
		#pragma endregion
	}

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

template <typename T>
std::chrono::microseconds Add(T* inputData, T* outputData, int height, int width, int channel)
{
	for (int i = 0; i < height * width * channel; ++i)
	{
		*outputData++ = *inputData++ + *outputData;
	}
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
	int size = height * width * channel;
	for (int i = size; i < size * 2; ++i)
	{
		inputData[i] = outputData[i - size];
	}
}
