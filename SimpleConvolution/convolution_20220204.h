#pragma once

#include <chrono>
#include "ops_20220208.h"
#include "utils.h"
#include <iostream>


std::chrono::microseconds _Convolution2D_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);

std::chrono::microseconds _Convolution2D_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);

std::chrono::microseconds _Convolution2D_Depthwise_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);

std::chrono::microseconds _Convolution2D_Depthwise_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);

std::chrono::microseconds _Convolution2D_Pointwise_k1_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);


std::chrono::microseconds _Convolution2D_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel = 3, int stride = 1, int padding = 1)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _Convolution2D_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel = 3, int stride = 2, int padding = 1)
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

	float* tempInputData1 = tempData;
	float* tempInputData2 = tempData + padWidth;
	float* tempInputData3 = tempInputData2 + padWidth;
	//float* tempOutputData = data;

	float weightVal[9];
	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		int outKernel = outCh * kernelSize;
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			float* val = data;
			//if (inCh == 0)
			//	*val += *bias;

			int kernelIndex = outKernel + inCh * 9;

			float weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
			float weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
			float weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

			//float weightVal_1 = *weight++, weightVal_2 = *weight++, weightVal_3 = *weight++;
			//float weightVal_4 = *weight++, weightVal_5 = *weight++, weightVal_6 = *weight++;
			//float weightVal_7 = *weight++, weightVal_8 = *weight++, weightVal_9 = *weight++;
			//memcpy(weightVal, weight, 4 * 9);
			//weight += 9;
			for (int row = 0; row < height; row += stride)
			{
				for (int col = 0; col < width; col += stride)
				{
					//float* val = data;

					float val1 = *(tempInputData1);
					float val2 = *(tempInputData1 + 1);
					float val3 = *(tempInputData1 + 2);

					float val4 = *(tempInputData2);
					float val5 = *(tempInputData2 + 1);
					float val6 = *(tempInputData2 + 2);

					float val7 = *(tempInputData3);
					float val8 = *(tempInputData3 + 1);
					float val9 = *(tempInputData3 + 2);

					*val += val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
						val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
						val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9;

					//*val += *(tempInputData1++) * weightVal_1 + *(tempInputData1++) * weightVal_2 + *(tempInputData1)*weightVal_3 +
					//	*(tempInputData2++) * weightVal_4 + *(tempInputData2++) * weightVal_5 + *(tempInputData2)*weightVal_6 +
					//	*(tempInputData3++) * weightVal_7 + *(tempInputData3++) * weightVal_8 + *(tempInputData3)*weightVal_9;

					//*val += *(tempInputData1++) * weightVal[0] + *(tempInputData1++) * weightVal[1] + *(tempInputData1)*weightVal[2] +
					//	*(tempInputData2++) *  weightVal[3] + *(tempInputData2++) *  weightVal[4] + *(tempInputData2)* weightVal[5] +
					//	*(tempInputData3++) *  weightVal[6] + *(tempInputData3++) *  weightVal[7] + *(tempInputData3)* weightVal[8];
					//*data = val;

					tempInputData1 += stride;
					tempInputData2 += stride;
					tempInputData3 += stride;
					++val;
				}
				tempInputData1 += padWidth + 2;
				tempInputData2 += padWidth + 2;
				tempInputData3 += padWidth + 2;
			}
			tempInputData1 += padWidth * 2;
			tempInputData2 += padWidth * 2;
			tempInputData3 += padWidth * 2;
			//data -= outputArea;
		}
		for (int i = 0; i < outputArea; ++i)
		{
			float val = *data + *bias;
			//val = (val < 0) ? 0 : val;
			*data = val;
			++data;
		}
		++bias;
		//data += outputArea;
		//weight -= inChannel * 9;
		tempInputData1 = tempData;
		tempInputData2 = tempInputData1 + padWidth;
		tempInputData3 = tempInputData2 + padWidth;
	}

	tensor->height = outputHeight;
	tensor->width = outputWidth;
	tensor->channel = outChannel;
	tensor->data = saveDataPos;
	//tempData = saveTempPos;

	delete[] saveTempPos;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _Convolution2D_Depthwise_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel = 3, int stride = 1, int padding = 1)
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

	float* tempInputData1 = tempData;
	float* tempInputData2 = tempData + padWidth;
	float* tempInputData3 = tempInputData2 + padWidth;
	//float* tempOutputData = data;

	float* val = data;

	for (int inCh = 0; inCh < inChannel; ++inCh)
	{
		int kernelIndex = inCh * 9;

		float weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
		float weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
		float weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

		//float weightVal_1 = *weight++, weightVal_2 = *weight++, weightVal_3 = *weight++;
		//float weightVal_4 = *weight++, weightVal_5 = *weight++, weightVal_6 = *weight++;
		//float weightVal_7 = *weight++, weightVal_8 = *weight++, weightVal_9 = *weight++;

		//int inChPos = inCh * height * width;
		for (int row = 0; row < height; ++row)
		{
			//int rowPos_1 = inChPos + row * width;
			//int rowPos_2 = rowPos_1 + width;
			//int rowPos_3 = rowPos_2 + width;
			for (int col = 0; col < width; ++col)
			{
				//int colPos_1 = rowPos_1 + col;
				//int colPos_2 = rowPos_2 + col;
				//int colPos_3 = rowPos_3 + col;

				//float val1 = *(tempInputData1);
				//float val2 = *(tempInputData1 + 1);
				//float val3 = *(tempInputData1 + 2);

				//float val4 = *(tempInputData2);
				//float val5 = *(tempInputData2 + 1);
				//float val6 = *(tempInputData2 + 2);

				//float val7 = *(tempInputData3);
				//float val8 = *(tempInputData3 + 1);
				//float val9 = *(tempInputData3 + 2);

				//*val += val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
				//	val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
				//	val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9 + *bias;


				//*val += *(tempInputData1++) * weightVal_1 + *(tempInputData1)*weightVal_2 + *(tempInputData1 + 1) * weightVal_3 +
				//	*(tempInputData2++) * weightVal_4 + *(tempInputData2)*weightVal_5 + *(tempInputData2 + 1) * weightVal_6 +
				//	*(tempInputData3++) * weightVal_7 + *(tempInputData3)*weightVal_8 + *(tempInputData3 + 1) * weightVal_9 + *bias;

				*val += *(tempInputData1) * weightVal_1 + *(tempInputData1+1)*weightVal_2 + *(tempInputData1 + 2) * weightVal_3 +
					*(tempInputData2) * weightVal_4 + *(tempInputData2+1)*weightVal_5 + *(tempInputData2 + 2) * weightVal_6 +
					*(tempInputData3) * weightVal_7 + *(tempInputData3+1)*weightVal_8 + *(tempInputData3 + 2) * weightVal_9 + *bias;

				//float val1 = tempData[colPos_1];
				//float val2 = tempData[colPos_1+1];
				//float val3 = tempData[colPos_1+2];
				//			 
				//float val4 = tempData[colPos_2];
				//float val5 = tempData[colPos_2+1];
				//float val6 = tempData[colPos_2+2];
				//			 
				//float val7 = tempData[colPos_3];
				//float val8 = tempData[colPos_3+1];
				//float val9 = tempData[colPos_3+2];

				//*val += val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
				//	val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
				//	val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9 + *bias;



				//*data = val;

				++tempInputData1;
				++tempInputData2;
				++tempInputData3;
				++val;
			}
			tempInputData1 += 2;
			tempInputData2 += 2;
			tempInputData3 += 2;
		}
		tempInputData1 += padWidth * 2;
		tempInputData2 += padWidth * 2;
		tempInputData3 += padWidth * 2;

		//data = tempOutputData;
		//for (int i = 0; i < area; ++i)
		//{
		//	float val = *data + *bias;
			//val = (val < 0) ? 0 : val;
		//	*data = val;

		//	++data;
		//}
		++bias;
		//tempOutputData = data;
	}
	tensor->height = outputHeight;
	tensor->width = outputWidth;
	//tensor->data = saveDataPos;
	//tempData = saveTempPos;
	delete[] saveTempPos;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _Convolution2D_Depthwise_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel = 3, int stride = 2, int padding = 1)
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

	float* tempInputData1 = tempData;
	float* tempInputData2 = tempData + padWidth;
	float* tempInputData3 = tempData + padWidth + padWidth;
	float* tempOutputData = data;

	float* val = data;
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
				//float val1 = *(tempInputData1);
				//float val2 = *(tempInputData1 + 1);
				//float val3 = *(tempInputData1 + 2);
				//
				//float val4 = *(tempInputData2);
				//float val5 = *(tempInputData2 + 1);
				//float val6 = *(tempInputData2 + 2);
				//
				//float val7 = *(tempInputData3);
				//float val8 = *(tempInputData3 + 1);
				//float val9 = *(tempInputData3 + 2);

				//val = val + val1 * weightVal_1 + val2 * weightVal_2 + val3 * weightVal_3 +
				//	val4 * weightVal_4 + val5 * weightVal_5 + val6 * weightVal_6 +
				//	val7 * weightVal_7 + val8 * weightVal_8 + val9 * weightVal_9;
				*val += *(tempInputData1)*weightVal_1 + *(tempInputData1 + 1) * weightVal_2 + *(tempInputData1 + 2) * weightVal_3 +
					*(tempInputData2)*weightVal_4 + *(tempInputData2 + 1) * weightVal_5 + *(tempInputData2 + 2) * weightVal_6 +
					*(tempInputData3)*weightVal_7 + *(tempInputData3 + 1) * weightVal_8 + *(tempInputData3 + 2) * weightVal_9 + *bias;

				//*data = val;
				tempInputData1 += stride;
				tempInputData2 += stride;
				tempInputData3 += stride;
				++val;
			}
			tempInputData1 += padWidth + 2;
			tempInputData2 += padWidth + 2;
			tempInputData3 += padWidth + 2;
		}
		tempInputData1 += padWidth * 2;
		tempInputData2 += padWidth * 2;
		tempInputData3 += padWidth * 2;

		//data = tempOutputData;
		//for (int i = 0; i < area; ++i)
		//{
		//	float val = *data + *bias;
		////	val = (val < 0) ? 0 : val;
		//	*data = val;

		//	++data;
		//}
		++bias;
		//tempOutputData = data;
	}
	tensor->height = outputHeight;
	tensor->width = outputWidth;
	//tensor->data = saveDataPos;
	//tempData = saveTempPos;
	delete[] saveTempPos;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

std::chrono::microseconds _Convolution2D_Pointwise_k1_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel = 1, int stride = 1, int padding = 0)
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

	float* o = data;
	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		float* d = o;
		float* v = tempData;
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			o = d;
			float weightVal = *weight;

			if (inCh == 0)
			{
				for (int i = 0; i < area; ++i)
				{
					(*o) += *v * weightVal + *bias;
					++o;
					++v;
				}
			}
			else
			{
				for (int i = 0; i < area; ++i)
				{
					(*o) += *v * weightVal;
					++o;
					++v;
				}
			}
			//	*o += *bias;
			//for (int i = 0; i < area; ++i)
			//{
			//	(*o) += *v * weightVal;
			//	++o;
			//	++v;
			//}
			++weight;
		}
		//data = o;
		//tempData = saveTempPos;
		++bias;
	}
	tensor->channel = outChannel;
	//tensor->data = saveDataPos;
	delete[] tempData;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}



std::chrono::microseconds Transpose(Tensor* tensor, int f);
std::chrono::microseconds Transpose(Tensor* tensor, int f)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;

	float* data = tensor->data;
	float* transTensor = new float[height * width * channel];
	memcpy(transTensor, data, sizeof(float) * height * width * channel);

	int idx = 0;
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			for (int ch = 0; ch < channel; ++ch, ++idx)
			{
				data[idx] = transTensor[ch * height * width + row * width + col];
			}
		}
	}
	delete[] transTensor;

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}














void Transpose(Tensor* tensor);
void Transpose(Tensor* tensor)
{
	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;

	float* data = tensor->data;
	float* transTensor = new float[height * width * channel];
	memcpy(transTensor, data, sizeof(float) * height * width * channel);

	int idx = 0;
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			for (int ch = 0; ch < channel; ++ch, ++idx)
			{
				data[idx] = transTensor[ch * height * width + row * width + col];
			}
		}
	}
	delete[] transTensor;

}

void Mult(Tensor* tensor, Tensor* _tensor);
void Mult(Tensor* tensor, Tensor* _tensor)
{
	int size = tensor->height * tensor->width * tensor->channel;
	float* data = tensor->data;
	float* _data = _tensor->data;

	for (int i = 0; i < size; ++i)
	{
		data[i] = data[i] * _data[i];
	}
}

void Equal(Tensor* tensor, Tensor* _tensor);
void Equal(Tensor* tensor, Tensor* _tensor)
{
	int size = tensor->height * tensor->width * tensor->channel;
	float* data = tensor->data;
	float* _data = _tensor->data;

	for (int i = 0; i < size; ++i)
	{
		data[i] = (data[i] == _data[i]) ? 1 : 0;
	}
}


struct topk
{
	int index;
	float value;
	topk(int _index, float _value) :index(_index), value(_value) {}
	bool operator<(const topk t) const { return this->value < t.value; }
	bool operator>(const topk t) const { return this->value > t.value; }

	bool operator<(const float t) const { return this->value < t; }
	bool operator>(const float t) const { return this->value > t; }
};
struct topkGreater
{
	bool operator()(topk a, topk b)
	{
		return a.value > b.value;
	}
};
struct topkLess
{
	bool operator()(topk a, topk b)
	{
		return a.value < b.value;
	}
};



std::vector<topk> TopK(Tensor* tensor, int k);

std::vector<topk> TopK(Tensor* tensor, int k)
{
	int size = tensor->height * tensor->width * tensor->channel;
	float* data = tensor->data;

	std::priority_queue<topk, std::vector<topk>, topkGreater> prique;
	for (int i = 0; i < size; ++i)
	{
		if (prique.size() < k)
		{
			prique.push(topk{ i, data[i] });
		}
		else if (prique.top().value < data[i])
		{
			prique.pop();
			prique.push(topk{ i, data[i] });
		}
	}

	std::vector<topk> outputs;
	outputs.reserve(k);
	while (!prique.empty())
	{
		outputs.push_back(prique.top());
		prique.pop();
	}
	return outputs;
}




struct Detection
{
	int category;
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
};

std::vector<Detection> Postprocessing(Tensor* offset, Tensor* size, Tensor* keypoint);
std::vector<Detection> Postprocessing(Tensor* offset, Tensor* size, Tensor* keypoint)
{
	int classNum = 2;
	int k = 10;
	int tensorDim = keypoint->height;

	Transpose(offset);
	Transpose(size);
	float* offsetData = offset->data;
	float* sizeData = size->data;

	_Sigmoid(keypoint);

	Tensor tempTensor;
	tempTensor.data = new float[tensorDim * tensorDim * 64];

	CopyTensor(&tempTensor, keypoint);
	_MaxPool(&tempTensor, 3, 1, 1);

	Equal(&tempTensor, keypoint);

	Mult(keypoint, &tempTensor);
	Transpose(keypoint);
	delete[] tempTensor.data;

	std::vector<topk> topkOutput = TopK(keypoint, k);

	int tempIndices[10];
	int yIndices[10];
	int xIndices[10];
	int detectionClasses[10];

	for (int i = 0; i < k; ++i)
	{
		tempIndices[i] = topkOutput[i].index / classNum;

		yIndices[i] = tempIndices[i] / tensorDim;
		xIndices[i] = tempIndices[i] - (yIndices[i] * tensorDim);
		detectionClasses[i] = topkOutput[i].index - (tempIndices[i] * classNum);
	}

	float sizes[20];
	float offsets[20];
	for (int i = 0; i < k; ++i)
	{
		sizes[i] = sizeData[yIndices[i] * tensorDim * 2 + xIndices[i] * 2 + 0] / 2;
		sizes[i + 10] = sizeData[yIndices[i] * tensorDim * 2 + xIndices[i] * 2 + 1] / 2;

		offsets[i] = offsetData[yIndices[i] * tensorDim * 2 + xIndices[i] * 2 + 0];
		offsets[i + 10] = offsetData[yIndices[i] * tensorDim * 2 + xIndices[i] * 2 + 1];
	}

	float pos[20];
	for (int i = 0; i < k; ++i)
	{
		pos[i] = yIndices[i] + offsets[i];
		pos[i + 10] = xIndices[i] + offsets[i + 10];
	}

	float minPos[20];
	float maxPos[20];
	for (int i = 0; i < k; ++i)
	{
		minPos[i] = (pos[i] - sizes[i]) * 4;
		minPos[i + 10] = (pos[i + 10] - sizes[i + 10]) * 4;
		maxPos[i] = (pos[i] + sizes[i]) * 4;
		maxPos[i + 10] = (pos[i + 10] + sizes[i + 10]) * 4;
	}



	std::vector<Detection> result;
	for (int i = 0; i < k; ++i)
	{
		result.push_back(Detection{ detectionClasses[i], topkOutput[i].value, (int)minPos[i + 10], (int)minPos[i], (int)maxPos[i + 10], (int)maxPos[i] });
	}
	return result;
}
