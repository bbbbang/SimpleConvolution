#include "ops_c.h"
#include "utils_c.h"


void _Convolution2D_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);

void _Convolution2D_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);

void _Convolution2D_Depthwise_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);

void _Convolution2D_Depthwise_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);

void _Convolution2D_Pointwise_k1_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);


void _Convolution2D_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
	std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

	std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
}

void _Convolution2D_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
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
	float* tempData = (float*)malloc(sizeof(float) * inChannel * padArea);
	memcpy(tempData, data, sizeof(float) * inChannel * padArea);
	memset(data, 0, sizeof(float) * outChannel * outputArea);
	float* tempDataAddr = tempData;
	float* dataAddr = data;

	int kernelSize = inChannel * 9;

	float* tempInputData1 = tempData;
	float* tempInputData2 = tempData + padWidth;
	float* tempInputData3 = tempInputData2 + padWidth;

	float weightVal[9];
	for (int outCh = 0; outCh < outChannel; ++outCh)
	{
		int outKernel = outCh * kernelSize;
		for (int inCh = 0; inCh < inChannel; ++inCh)
		{
			float* val = data;
			int kernelIndex = outKernel + inCh * 9;

			float weightVal_1 = weight[kernelIndex + 0], weightVal_2 = weight[kernelIndex + 1], weightVal_3 = weight[kernelIndex + 2];
			float weightVal_4 = weight[kernelIndex + 3], weightVal_5 = weight[kernelIndex + 4], weightVal_6 = weight[kernelIndex + 5];
			float weightVal_7 = weight[kernelIndex + 6], weightVal_8 = weight[kernelIndex + 7], weightVal_9 = weight[kernelIndex + 8];

			for (int row = 0; row < height; row += stride)
			{
				for (int col = 0; col < width; col += stride)
				{
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
		}
		for (int i = 0; i < outputArea; ++i)
		{
			float val = *data + *bias;
			*data = val;
			++data;
		}
		++bias;
		tempInputData1 = tempData;
		tempInputData2 = tempInputData1 + padWidth;
		tempInputData3 = tempInputData2 + padWidth;
	}

	tensor->height = outputHeight;
	tensor->width = outputWidth;
	tensor->channel = outChannel;
	tensor->data = dataAddr;
	free(tempDataAddr);
	tempDataAddr = NULL;
}

void _Convolution2D_Depthwise_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
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
	float* tempData = (float*)malloc(sizeof(float) * inChannel * padArea);
	memcpy(tempData, data, sizeof(float) * inChannel * padArea);
	memset(data, 0, sizeof(float) * outChannel * outputArea);
	float* tempDataAddr = tempData;

	float* tempInputData1 = tempData;
	float* tempInputData2 = tempData + padWidth;
	float* tempInputData3 = tempInputData2 + padWidth;

	float* val = data;

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
				* val += *(tempInputData1)*weightVal_1 + *(tempInputData1 + 1) * weightVal_2 + *(tempInputData1 + 2) * weightVal_3 +
					*(tempInputData2)*weightVal_4 + *(tempInputData2 + 1) * weightVal_5 + *(tempInputData2 + 2) * weightVal_6 +
					*(tempInputData3)*weightVal_7 + *(tempInputData3 + 1) * weightVal_8 + *(tempInputData3 + 2) * weightVal_9 + *bias;

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

		++bias;
	}
	tensor->height = outputHeight;
	tensor->width = outputWidth;
	free(tempDataAddr);
	tempDataAddr = NULL;
}

void _Convolution2D_Depthwise_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
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
	float* tempData = (float*)malloc(sizeof(inChannel) * padArea);
	memcpy(tempData, data, sizeof(float) * inChannel * padArea);
	memset(data, 0, sizeof(float) * outChannel * outputArea);
	float* tempDataAddr = tempData;

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
				*val += *(tempInputData1)*weightVal_1 + *(tempInputData1 + 1) * weightVal_2 + *(tempInputData1 + 2) * weightVal_3 +
					*(tempInputData2)*weightVal_4 + *(tempInputData2 + 1) * weightVal_5 + *(tempInputData2 + 2) * weightVal_6 +
					*(tempInputData3)*weightVal_7 + *(tempInputData3 + 1) * weightVal_8 + *(tempInputData3 + 2) * weightVal_9 + *bias;

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

		++bias;
	}
	tensor->height = outputHeight;
	tensor->width = outputWidth;
	free(tempDataAddr);
	tempDataAddr = NULL;

}

void _Convolution2D_Pointwise_k1_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;
	int area = height * width;

	float* data = tensor->data;
	float* tempData = (float*)malloc(sizeof(float) * inChannel * area);
	memcpy(tempData, data, sizeof(float) * inChannel * area);
	memset(data, 0, sizeof(float) * outChannel * area);

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
			++weight;
		}
		++bias;
	}
	tensor->channel = outChannel;
	free(tempData);
	tempData = NULL;
}



void Transpose(Tensor* tensor, int f);
void Transpose(Tensor* tensor, int f)
{
	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;

	float* data = tensor->data;
	float* transTensor = (float*)malloc(sizeof(float) * height * width * channel);
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
	free(transTensor);
	transTensor = NULL;
}


void _Transpose(Tensor* tensor);
void _Transpose(Tensor* tensor)
{
	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;

	float* data = tensor->data;
	float* transTensor = (float*)malloc(sizeof(float) * height * width * channel);
	memcpy(transTensor, data, sizeof(float) * height * width * channel);

	int idx = 0;
	for (int ch = 0; ch < channel; ++ch)
	{
		for (int row = 0; row < height; ++row)
		{
			for (int col = 0; col < width; ++col)
			{
				data[idx] = transTensor[row * width * channel + col * channel + ch];
				++idx;
			}
		}
	}
	free(transTensor);
	transTensor = NULL;
}


void Transpose(Tensor* tensor);
void Transpose(Tensor* tensor)
{
	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;

	float* data = tensor->data;
	float* transTensor = (float*)malloc(sizeof(float) * height * width * channel);
	memcpy(transTensor, data, sizeof(float) * height * width * channel);

	int idx = 0;
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			for (int ch = 0; ch < channel; ++ch)
			{
				data[idx] = transTensor[ch * height * width + row * width + col];
				++idx;
			}
		}
	}
	free(transTensor);
	transTensor = NULL;
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


//struct topk
//{
//	int index;
//	float value;
//	topk(int _index, float _value) :index(_index), value(_value) {}
//	bool operator<(const topk t) const { return this->value < t.value; }
//	bool operator>(const topk t) const { return this->value > t.value; }
//
//	bool operator<(const float t) const { return this->value < t; }
//	bool operator>(const float t) const { return this->value > t; }
//};
//struct topkGreater
//{
//	bool operator()(topk a, topk b)
//	{
//		return a.value > b.value;
//	}
//};
//struct topkLess
//{
//	bool operator()(topk a, topk b)
//	{
//		return a.value < b.value;
//	}
//};
//
//
//
//std::vector<topk> TopK(Tensor* tensor, int k);
//
//std::vector<topk> TopK(Tensor* tensor, int k)
//{
//	int size = tensor->height * tensor->width * tensor->channel;
//	float* data = tensor->data;
//
//	std::priority_queue<topk, std::vector<topk>, topkGreater> prique;
//	for (int i = 0; i < size; ++i)
//	{
//		if (prique.size() < k)
//		{
//			prique.push(topk{ i, data[i] });
//		}
//		else if (prique.top().value < data[i])
//		{
//			prique.pop();
//			prique.push(topk{ i, data[i] });
//		}
//	}
//
//	std::vector<topk> outputs;
//	outputs.reserve(k);
//	while (!prique.empty())
//	{
//		outputs.push_back(prique.top());
//		prique.pop();
//	}
//	return outputs;
//}
//
//
//
//
//typedef struct _Detection
//{
//	int category;
//	float score;
//	int x1;
//	int y1;
//	int x2;
//	int y2;
//}Detection;
//
//std::vector<Detection> Postprocessing(Tensor* offset, Tensor* size, Tensor* keypoint);
//std::vector<Detection> Postprocessing(Tensor* offset, Tensor* size, Tensor* keypoint)
//{
//	int classNum = 2;
//	int k = 10;
//	int tensorDim = keypoint->height;
//
//	Transpose(offset);
//	Transpose(size);
//	float* offsetData = offset->data;
//	float* sizeData = size->data;
//
//	_Sigmoid(keypoint);
//
//	Tensor tempTensor;
//	tempTensor.data = new float[tensorDim * tensorDim * 64];
//
//	CopyTensor(&tempTensor, keypoint);
//	_MaxPool(&tempTensor, 3, 1, 1);
//
//	Equal(&tempTensor, keypoint);
//
//	Mult(keypoint, &tempTensor);
//	Transpose(keypoint);
//	delete[] tempTensor.data;
//
//	std::vector<topk> topkOutput = TopK(keypoint, k);
//
//	int tempIndices[10];
//	int yIndices[10];
//	int xIndices[10];
//	int detectionClasses[10];
//
//	for (int i = 0; i < k; ++i)
//	{
//		tempIndices[i] = topkOutput[i].index / classNum;
//
//		yIndices[i] = tempIndices[i] / tensorDim;
//		xIndices[i] = tempIndices[i] - (yIndices[i] * tensorDim);
//		detectionClasses[i] = topkOutput[i].index - (tempIndices[i] * classNum);
//	}
//
//	float sizes[20];
//	float offsets[20];
//	for (int i = 0; i < k; ++i)
//	{
//		sizes[i] = sizeData[yIndices[i] * tensorDim * 2 + xIndices[i] * 2 + 1] / 2;
//		sizes[i + 10] = sizeData[yIndices[i] * tensorDim * 2 + xIndices[i] * 2 + 0] / 2;
//
//		offsets[i] = offsetData[yIndices[i] * tensorDim * 2 + xIndices[i] * 2 + 1];
//		offsets[i + 10] = offsetData[yIndices[i] * tensorDim * 2 + xIndices[i] * 2 + 0];
//	}
//
//	float pos[20];
//	for (int i = 0; i < k; ++i)
//	{
//		pos[i] = yIndices[i] + offsets[i];
//		pos[i + 10] = xIndices[i] + offsets[i + 10];
//	}
//
//	float minPos[20];
//	float maxPos[20];
//	for (int i = 0; i < k; ++i)
//	{
//		minPos[i] = (pos[i] - sizes[i]) * 4;
//		minPos[i + 10] = (pos[i + 10] - sizes[i + 10]) * 4;
//		maxPos[i] = (pos[i] + sizes[i]) * 4;
//		maxPos[i + 10] = (pos[i + 10] + sizes[i + 10]) * 4;
//	}
//
//	std::vector<Detection> result;
//	for (int i = 0; i < k; ++i)
//	{
//		result.push_back(Detection{ detectionClasses[i], topkOutput[i].value, (int)minPos[i + 10], (int)minPos[i], (int)maxPos[i + 10], (int)maxPos[i] });
//	}
//	return result;
//}
