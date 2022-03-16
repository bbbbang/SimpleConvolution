#include "utils.h"



//void _Relu(Tensor* inputData);
//void _ZeroConcat(Tensor* inputData);
//void _Softmax(Tensor* inputData);
//void _ZeroPadding(Tensor* inputData, int padding);
//void _MaxPool(Tensor* inputData, int kernel, int stride, int padding);
//void _Resize(Tensor* inputData, float scale);
//void _Add(Tensor* inputData, Tensor* outputData);
//void _Concat(Tensor* inputData, Tensor* outputData);
//void _Sigmoid(Tensor* tensor);



// h w c -> c h w
void _Transpose(Tensor* tensor)
{
	size_t height = tensor->height;
	size_t width = tensor->width;
	size_t channel = tensor->channel;

	float* data = tensor->data;
	memcpy(tempTensor, data, sizeof(float) * height * width * channel);

	size_t idx = 0;
	for (size_t ch = 0; ch < channel; ++ch)
	{
		for (size_t row = 0; row < height; ++row)
		{
			for (size_t col = 0; col < width; ++col)
			{
				data[idx] = tempTensor[row * width * channel + col * channel + ch];
				++idx;
			}
		}
	}
	size_t area = height * width;
	for (size_t ch = 0; ch < channel; ++ch)
	{
		for (size_t i = 0; i < area; ++i)
		{
			tempTensor[i];
			tempTensor[i + channel];
		}
	}

	for (size_t i = 0; i < area; ++i)
	{
		for (size_t ch = 0; ch < channel; ++ch)
		{
			tempTensor[i];
			tempTensor[i + channel];
		}
	}
}

// c h w -> h w c
void Transpose(Tensor* tensor)
{
	size_t height = tensor->height;
	size_t width = tensor->width;
	size_t channel = tensor->channel;

	float* data = tensor->data;
	memcpy(tempTensor, data, sizeof(float) * height * width * channel);

	size_t idx = 0;
	for (size_t row = 0; row < height; ++row)
	{
		for (size_t col = 0; col < width; ++col)
		{
			for (size_t ch = 0; ch < channel; ++ch)
			{
				data[idx] = tempTensor[ch * height * width + row * width + col];
				++idx;
			}
		}
	}
}




void _Relu(Tensor* inputData)
{
	float* data = inputData->data;
	int size = inputData->width * inputData->height * inputData->channel;

	
	if (size % 10 == 0)
	{
		float ff[10];
		for (int i = 0; i < size; i += 10)
		{
			memcpy(ff, data, sizeof(float) * 10);
			//ff[0] = max(ff[0], 0);
			//ff[1] = max(ff[1], 0);
			//ff[2] = max(ff[2], 0);
			//ff[3] = max(ff[3], 0);
			//ff[4] = max(ff[4], 0);
			//ff[5] = max(ff[5], 0);
			//ff[6] = max(ff[6], 0);
			//ff[7] = max(ff[7], 0);
			//ff[8] = max(ff[8], 0);
			//ff[9] = max(ff[9], 0);

			ff[0] = (ff[0] < 0) ? 0 : ff[0];
			ff[1] = (ff[1] < 0) ? 0 : ff[1];
			ff[2] = (ff[2] < 0) ? 0 : ff[2];
			ff[3] = (ff[3] < 0) ? 0 : ff[3];
			ff[4] = (ff[4] < 0) ? 0 : ff[4];
			ff[5] = (ff[5] < 0) ? 0 : ff[5];
			ff[6] = (ff[6] < 0) ? 0 : ff[6];
			ff[7] = (ff[7] < 0) ? 0 : ff[7];
			ff[8] = (ff[8] < 0) ? 0 : ff[8];
			ff[9] = (ff[9] < 0) ? 0 : ff[9];
			memcpy(data, ff, sizeof(float) * 10);
			data += 10;

			//data[i] = max(data[i], 0);
			//data[i+1] = max(data[i+1], 0);
			//data[i+2] = max(data[i+2], 0);
			//data[i+3] = max(data[i+3], 0);
			//data[i+4] = max(data[i+4], 0);
			//data[i+5] = max(data[i+5], 0);
			//data[i+6] = max(data[i+6], 0);
			//data[i+7] = max(data[i+7], 0);
			//data[i+8] = max(data[i+8], 0);
			//data[i + 9] = max(data[i + 9], 0);

			//data[i] = (data[i] < 0) ? 0 : data[i];
			//data[i+1] = (data[i+1] < 0) ? 0 : data[i+1];
			//data[i+2] = (data[i+2] < 0) ? 0 : data[i+2];
			//data[i+3] = (data[i+3] < 0) ? 0 : data[i+3];
			//data[i+4] = (data[i+4] < 0) ? 0 : data[i+4];
			//data[i+5] = (data[i+5] < 0) ? 0 : data[i+5];
			//data[i+6] = (data[i+6] < 0) ? 0 : data[i+6];
			//data[i+7] = (data[i+7] < 0) ? 0 : data[i+7];
			//data[i+8] = (data[i+8] < 0) ? 0 : data[i+8];
			//data[i+9] = (data[i+9] < 0) ? 0 : data[i+9];
		}
	}
	else
	{
		float ff[5];
		for (int i = 0; i < size; i += 5)
		{
			memcpy(ff, data, sizeof(float) * 5);
			//ff[0] = max(ff[0], 0);
			//ff[1] = max(ff[1], 0);
			//ff[2] = max(ff[2], 0);
			//ff[3] = max(ff[3], 0);
			//ff[4] = max(ff[4], 0);
			//ff[5] = max(ff[5], 0);
			//ff[6] = max(ff[6], 0);
			//ff[7] = max(ff[7], 0);
			//ff[8] = max(ff[8], 0);
			//ff[9] = max(ff[9], 0);

			ff[0] = (ff[0] < 0) ? 0 : ff[0];
			ff[1] = (ff[1] < 0) ? 0 : ff[1];
			ff[2] = (ff[2] < 0) ? 0 : ff[2];
			ff[3] = (ff[3] < 0) ? 0 : ff[3];
			ff[4] = (ff[4] < 0) ? 0 : ff[4];
			memcpy(data, ff, sizeof(float) * 5);
			data += 5;

			//data[i] = max(data[i], 0);
			//data[i + 1] = max(data[i + 1], 0);
			//data[i + 2] = max(data[i + 2], 0);
			//data[i + 3] = max(data[i + 3], 0);
			//data[i + 4] = max(data[i + 4], 0);

			//data[i] = (data[i] < 0) ? 0 : data[i];
			//data[i + 1] = (data[i + 1] < 0) ? 0 : data[i + 1];
			//data[i + 2] = (data[i + 2] < 0) ? 0 : data[i + 2];
			//data[i + 3] = (data[i + 3] < 0) ? 0 : data[i + 3];
			//data[i + 4] = (data[i + 4] < 0) ? 0 : data[i + 4];
		}
	}
}
void _ZeroConcat(Tensor* inputData)
{
	float* data = inputData->data;
	int size = inputData->width * inputData->height * inputData->channel;

	for (int i = size; i < size * 2; ++i)
	{
		data[i] = 0;
	}
	inputData->channel = inputData->channel * 2;
}
void _Softmax(Tensor* inputData)
{
	int channel = inputData->channel;
	float sum = 0;
	float* data = inputData->data;

	for (int i = 0; i < channel; ++i)
	{
		float val = exp(data[i]);

		data[i] = val;
		sum += val;
	}
	for (int i = 0; i < channel; ++i)
	{
		data[i] = data[i] / sum;
	}
}

void _ZeroPadding(Tensor* inputData, int padding)
{
	int height = inputData->height;
	int width = inputData->width;
	int channel = inputData->channel;
	int area = height * width;

	int outputWidth = (height + padding * 2);
	int outputArea = outputWidth * outputWidth;

	float* data = inputData->data;
	//float* tempData = (float*)malloc(sizeof(float) * area * channel);
	memcpy(tempTensor, data, sizeof(float) * area * channel);
	memset(data, 0, sizeof(float) * outputArea * channel);
	float* tempDataAddr = tempTensor;
	float* dataAddr = data;

	for (int ch = 0; ch < channel; ++ch)
	{
		data += outputWidth;
		for (int row = 0; row < height; ++row)
		{
			++data;
			for (int col = 0; col < width; ++col)
			{
				*data = *tempDataAddr;
				++data;
				++tempDataAddr;
			}
			++data;
		}
		data += outputWidth;
	}
	inputData->height = outputWidth;
	inputData->width = outputWidth;
	inputData->data = dataAddr;
	//free(tempDataAddr);
	//tempDataAddr = NULL;
}
void _MaxPool(Tensor* inputData, int kernel, int stride, int padding)
{
	int height = inputData->height;
	int width = inputData->width;
	int channel = inputData->channel;
	int area = height * width;

	int padHeight = height + padding * 2;
	int padWidth = width + padding * 2;
	int tensorSize = padHeight * padWidth * channel;

	int outputHeight = (padHeight - kernel) / stride + 1;
	int outputWidth = (padWidth - kernel) / stride + 1;

	// padding
	if (padding > 0)
	{
		_ZeroPadding(inputData, padding);
	}

	float* data = inputData->data;
	//float* tempData = (float*)malloc(sizeof(float) * tensorSize);
	memcpy(tempTensor, data, sizeof(float) * tensorSize);
	//memset(data, 0, sizeof(float) * area * channel);
	float* tempDataAddr = tempTensor;
	float* dataAddr = data;

	if (kernel == 3 && stride == 1)
	{
		for (int ch = 0; ch < channel; ++ch)
		{
			for (int row = 0; row < height; ++row)
			{
				for (int col = 0; col < width; ++col)
				{
					float* pos1 = tempDataAddr;
					float* pos2 = tempDataAddr + padWidth;
					float* pos3 = pos2 + padWidth;

					float val1 = *pos1;
					float val2 = *(pos1 + 1);
					float val3 = *(pos1 + 2);
					float val4 = *pos2;
					float val5 = *(pos2 + 1);
					float val6 = *(pos2 + 2);
					float val7 = *pos3;
					float val8 = *(pos3 + 1);
					float val9 = *(pos3 + 2);
					float m = val1;
					m = val2 > m ? val2 : m;
					m = val3 > m ? val3 : m;
					m = val4 > m ? val4 : m;
					m = val5 > m ? val5 : m;
					m = val6 > m ? val6 : m;
					m = val7 > m ? val7 : m;
					m = val8 > m ? val8 : m;
					m = val9 > m ? val9 : m;
					*data = m;

					++data;
					++tempDataAddr;
				}
				tempDataAddr += 2;
			}
			tempDataAddr += padWidth * 2;
		}
	}
	else if (stride == 2)
	{
		for (int ch = 0; ch < channel; ++ch)
		{
			for (int row = 0; row < height; row += stride)
			{
				for (int col = 0; col < width; col += stride)
				{
					float val1 = *tempDataAddr;
					float val2 = *(tempDataAddr + 1);
					float val3 = *(tempDataAddr + padWidth);
					float val4 = *(tempDataAddr + padWidth + 1);
					float m = val1;
					m = val2 > m ? val2 : m;
					m = val3 > m ? val3 : m;
					m = val4 > m ? val4 : m;
					*data = m;

					++data;
					tempDataAddr += stride;
				}
				tempDataAddr += padWidth;
			}
		}
	}

	inputData->height = outputHeight;
	inputData->width = outputWidth;
	inputData->data = dataAddr;
	//free(tempDataAddr);
	//tempDataAddr = NULL;
}
void _Resize(Tensor* inputData, float scale)
{
	int height = inputData->height;
	int width = inputData->width;
	int channel = inputData->channel;
	int area = height * width;

	int outputHeight = height * scale;
	int outputWidth = width * scale;
	int outputArea = outputHeight * outputWidth;

	float* data = inputData->data;
	//float* tempData = (float*)malloc(sizeof(float) * area * channel);
	memcpy(tempTensor, data, sizeof(float) * area * channel);
	memset(data, 0, sizeof(float) * outputArea * channel);
	float* tempDataAddr = tempTensor;
	float* dataAddr = data;

	for (int ch = 0; ch < channel; ++ch)
	{
		for (int row = 0; row < height; ++row)
		{
			for (int col = 0; col < width; ++col)
			{
				float val = *tempDataAddr;

				*data = val;
				*(data + 1) = val;
				*(data + outputWidth) = val;
				*(data + outputWidth + 1) = val;

				data += 2;
				++tempDataAddr;
			}
			data += outputWidth;
		}
	}

	inputData->height = outputHeight;
	inputData->width = outputWidth;
	inputData->data = dataAddr;
	//free(tempDataAddr);
	//tempDataAddr = NULL;
}

void _Add(Tensor* inputData, Tensor* outputData)
{
	float* data = inputData->data;
	float* _data = outputData->data;

	int size = inputData->width * inputData->height * inputData->channel;

	if (size % 10 == 0)
	{
		for (int i = 0; i < size; i += 10)
		{
			_data[i] += data[i];
			_data[i + 1] += data[i + 1];
			_data[i + 2] += data[i + 2];
			_data[i + 3] += data[i + 3];
			_data[i + 4] += data[i + 4];
			_data[i + 5] += data[i + 5];
			_data[i + 6] += data[i + 6];
			_data[i + 7] += data[i + 7];
			_data[i + 8] += data[i + 8];
			_data[i + 9] += data[i + 9];
		}
	}
	else
	{
		for (int i = 0; i < size; i += 5)
		{
			_data[i] += data[i];
			_data[i + 1] += data[i + 1];
			_data[i + 2] += data[i + 2];
			_data[i + 3] += data[i + 3];
			_data[i + 4] += data[i + 4];
		}
	}



	//for (int i = 0; i < size; ++i)
	//{
	//	_data[i] += data[i];
	//}
}
void _Concat(Tensor* inputData, Tensor* outputData)
{
	float* data = inputData->data;
	float* _data = outputData->data;
	int size = inputData->width * inputData->height * inputData->channel;

	for (int i = size; i < size * 2; ++i)
	{
		data[i] = _data[i - size];
	}
	inputData->channel = inputData->channel * 2;
}

void _Sigmoid(Tensor* tensor)
{
	float* data = tensor->data;
	int size = tensor->height * tensor->width * tensor->channel;

	for (int i = 0; i < size; ++i)
	{
		data[i] = 1 / (1 + exp(-data[i]));
	}
}




void CopyTensor(Tensor* dst, Tensor* src)
{
	dst->height = src->height;
	dst->width = src->width;
	dst->channel = src->channel;
	memcpy(dst->data, src->data, sizeof(float) * dst->height * dst->width * dst->channel);
}
