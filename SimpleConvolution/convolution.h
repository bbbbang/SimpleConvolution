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
				for (int col = 0; col < width; col += stride)
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
				tempInputData1 += padWidth+2;
				tempInputData2 += padWidth+2;
				tempInputData3 += padWidth+2;
			}
			tempInputData1 += padWidth*2 ;
			tempInputData2 += padWidth*2 ;
			tempInputData3 += padWidth*2 ;
			outputData -= outputArea;
		}
		for (int i = 0; i < outputArea; ++i)
		{
			T val = *outputData + *bias;
			val = (val < 0) ? 0 : val;
			*outputData = val;
			++outputData;
		}
		++bias;
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
		outputData[i] = 0;
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
				*outputData = val;

				++tempInputData1;
				++tempInputData2;
				++tempInputData3;
				++outputData;
			}
			tempInputData1 += 2;
			tempInputData2 += 2;
			tempInputData3 += 2;
		}
		tempInputData1 += padWidth*2;
		tempInputData2 += padWidth*2;
		tempInputData3 += padWidth*2;
		
		outputData = tempOutputData;
		for (int i = 0; i < area; ++i)
		{
			T val = *outputData + *bias;
			//val = (val < 0) ? 0 : val;
			*outputData = val;

			++outputData;
		}
		++bias;
		tempOutputData = outputData;
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
std::chrono::microseconds Transpose(T* inputData, T* outputData, int height, int width, int channel);

template <typename T>
std::chrono::microseconds Transpose(T* inputData, T* outputData, int height, int width, int channel)
{
	
}



template <typename T>
std::chrono::microseconds Convolution2D_Pointwise_k1_s1(T* inputData, T* outputData, T* weight, T* bias,
	int height, int width, int inChannel, int outChannel,
	int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int area = height * width;

	for (int i = 0; i < inChannel * area; ++i)
	{
		outputData[i] = 0;
	}

	T* tempOutputData = outputData;
	T* tempInputData = inputData;

	for (int outCh = 0; outCh < outChannel; ++outCh)
	{

	//	inputData = tempInputData;
	//	for (int i = 0; i < area; ++i)
	//	{
	//		T val = 0;
	//		for (int inCh = 0; inCh < inChannel; ++inCh)
	//		{
	//			
	//			T v = inputData[i + inCh];
	//			T f = *weight;
	//			//v = v * f;
	//			val += v * f;
	//		}
	//		val = val + *bias;
	//		val = (val < 0) ? 0 : val;
	//		*(outputData++) = val;
	//	}
	//	++bias;


		//int outPos = outCh * area;
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			T weightVal = *weight;
			//outputData = tempOutputData;
			int inPos = inCh * area;
			
			for (int i = 0; i < area; ++i)
			{
				T o = *outputData;
				//T o = outputData[outPos + i];
				//T p = *inputData++;
				//o = o + p * weightVal;
				o = o + inputData[inPos + i] * weightVal;
				//outputData[outPos + i] = o;
				(*outputData) = o;
				++outputData;
			}
			++weight;
			outputData -= area;
		}
		//outputData = tempOutputData;
		//inputData = tempInputData;

		for (int i = 0; i < area; ++i)
		{
			T val = *outputData + *bias;
			//val = (val < 0) ? 0 : val;
			*outputData = val;
			++outputData;
		}
		//tempOutputData = outputData;
		//std::cout << idx << std::endl;
		++bias;
		//++weight;
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
