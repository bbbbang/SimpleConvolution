#include "ops.h"


//void _Convolution2D_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);
//void _Convolution2D_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);
//void _Convolution2D_Depthwise_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);
//void _Convolution2D_Depthwise_k3_s2(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);
//void _Convolution2D_Pointwise_k1_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding);


void _Convolution2D_k3_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
	int a = 1;
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
	//float* tempData = (float*)malloc(sizeof(float) * inChannel * padArea);
	memcpy(tempTensor, data, sizeof(float) * inChannel * padArea);
	memset(data, 0, sizeof(float) * outChannel * outputArea);
	//float* tempDataAddr = tempData;
	float* dataAddr = data;

	int kernelSize = inChannel * 9;

	float* tempInputData1 = tempTensor;
	float* tempInputData2 = tempTensor + padWidth;
	float* tempInputData3 = tempInputData2 + padWidth;

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
		tempInputData1 = tempTensor;
		tempInputData2 = tempInputData1 + padWidth;
		tempInputData3 = tempInputData2 + padWidth;
	}

	tensor->height = outputHeight;
	tensor->width = outputWidth;
	tensor->channel = outChannel;
	tensor->data = dataAddr;

	//tempInputData1 = NULL;
	//tempInputData2 = NULL;
	//tempInputData3 = NULL;
	//free(tempData);
	//tempDataAddr = NULL;
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
	//float* tempData = (float*)malloc(sizeof(float) * inChannel * padArea);
	memcpy(tempTensor, data, sizeof(float) * inChannel * padArea);
	memset(data, 0, sizeof(float) * outChannel * outputArea);
	//float* tempDataAddr = tempData;

	float* tempInputData1 = tempTensor;
	float* tempInputData2 = tempTensor + padWidth;
	float* tempInputData3 = tempInputData2 + padWidth;

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
				*data = *(tempInputData1)*weightVal_1 + *(tempInputData1 + 1) * weightVal_2 + *(tempInputData1 + 2) * weightVal_3 +
					*(tempInputData2)*weightVal_4 + *(tempInputData2 + 1) * weightVal_5 + *(tempInputData2 + 2) * weightVal_6 +
					*(tempInputData3)*weightVal_7 + *(tempInputData3 + 1) * weightVal_8 + *(tempInputData3 + 2) * weightVal_9 + *bias;

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

		++bias;
	}
	tensor->height = outputHeight;
	tensor->width = outputWidth;

	//tempInputData1 = NULL;
	//tempInputData2 = NULL;
	//tempInputData3 = NULL;
	//free(tempData);
	//tempDataAddr = NULL;
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
	//float* tempData = (float*)malloc(sizeof(float) * inChannel * padArea);
	memcpy(tempTensor, data, sizeof(float) * inChannel * padArea);
	memset(data, 0, sizeof(float) * outChannel * outputArea);
	//float* tempDataAddr = tempTensor;

	float* tempInputData1 = tempTensor;
	float* tempInputData2 = tempTensor + padWidth;
	float* tempInputData3 = tempTensor + padWidth + padWidth;

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
				*data = *(tempInputData1)*weightVal_1 + *(tempInputData1 + 1) * weightVal_2 + *(tempInputData1 + 2) * weightVal_3 +
					*(tempInputData2)*weightVal_4 + *(tempInputData2 + 1) * weightVal_5 + *(tempInputData2 + 2) * weightVal_6 +
					*(tempInputData3)*weightVal_7 + *(tempInputData3 + 1) * weightVal_8 + *(tempInputData3 + 2) * weightVal_9 + *bias;

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

		++bias;
	}
	tensor->height = outputHeight;
	tensor->width = outputWidth;

	//val = NULL;
	//tempInputData1 = NULL;
	//tempInputData2 = NULL;
	//tempInputData3 = NULL;
	//free(tempData);
	//tempDataAddr = NULL;

}

void _Convolution2D_Pointwise_k1_s1(Tensor* tensor, float* weight, float* bias, int inChannel, int outChannel, int kernel, int stride, int padding)
{
	int height = tensor->height;
	int width = tensor->width;
	int channel = tensor->channel;
	int area = height * width;

	float* data = tensor->data;
	memcpy(tempTensor, data, sizeof(float) * inChannel * area);
	memset(data, 0, sizeof(float) * outChannel * area);


	Transpose(tensor);

	if (area % 10 == 0)
	{
		for (int outCh = 0; outCh < outChannel; ++outCh)
		{
			float* d = tempTensor;
			for (int i = 0; i < area; i += 10)
			{
				float val1 = 0; float val2 = 0; float val3 = 0;
				float val4 = 0; float val5 = 0; float val6 = 0;
				float val7 = 0; float val8 = 0; float val9 = 0; float val10 = 0;
				float* w = weight;
				float dd[10];
				float ww[10];
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val1 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val1 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

					d += 8;
					w += 8;
				}
				w = weight;

				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val2 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val2 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val3 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val3 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val4 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val4 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val5 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val5 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);
					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val6 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val6 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);
					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val7 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val7 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);
					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val8 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val8 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);
					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val9 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val9 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);
					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val10 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val10 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);
					d += 8;
					w += 8;
				}
				float b = *bias;
				float val[10] = { val1 + b,val2 + b,val3 + b,val4 + b,val5 + b,val6 + b,val7 + b,val8 + b,val9 + b,val10 + b };
				memcpy(data, val, sizeof(float) * 10);
				data += 10;
			}
			weight += inChannel;
			++bias;
		}
	}
	else
	{
		for (int outCh = 0; outCh < outChannel; ++outCh)
		{
			float* d = tempTensor;
			for (int i = 0; i < area; i += 5)
			{
				float val1 = 0;
				float val2 = 0;
				float val3 = 0;
				float val4 = 0;
				float val5 = 0;
				float* w = weight;

				float dd[8];
				float ww[8];
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val1 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val1 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

					d += 8;
					w += 8;
				}
				w = weight;

				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val2 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val2 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val3 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val3 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val4 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val4 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

					d += 8;
					w += 8;
				}
				w = weight;
				for (int inCh = 0; inCh < inChannel; inCh += 8)
				{
					//memcpy(dd, d, sizeof(4) * 8);
					//memcpy(ww, w, sizeof(4) * 8);
					//val5 += dd[0] * ww[0] + dd[1] * ww[1] + dd[2] * ww[2] + dd[3] * ww[3] +
					//	dd[4] * ww[4] + dd[5] * ww[5] + dd[6] * ww[6] + dd[7] * ww[7];

					val5 += *(d) * *(w)+*(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
						*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

					d += 8;
					w += 8;
				}
				float b = *bias;
				float val[5] = { val1 + b,val2 + b,val3 + b,val4 + b,val5 + b };
				memcpy(data, val, sizeof(float) * 5);
				data += 5;
			}
			weight += inChannel;
			++bias;
		}
	}







	//for (int outCh = 0; outCh < outChannel; ++outCh)
	//{
	//	float *d = tempTensor;
	//	//for (int i = 0; i < area; ++i)
	//	//{
	//	//	float val = 0;
	//	//	float* w = weight;


	//	//	//for (int inCh = 0; inCh < inChannel; ++inCh)
	//	//	//{
	//	//	//	val += *(d) * *(w);

	//	//	//	++d;
	//	//	//	++w;
	//	//	//}

	//	//	for (int inCh = 0; inCh < inChannel; inCh+=8)
	//	//	{
	//	//		val += *(d) * *(w) + *(d+1) * *(w+1) + *(d+2) * *(w+2) + *(d+3) * *(w+3) +
	//	//			*(d+4) * *(w+4) + *(d+5) * *(w+5) + *(d+6) * *(w+6) + *(d+7) * *(w+7);

	//	//		d+=8;
	//	//		w+=8;
	//	//	}

	//	//	*data = val + *bias;
	//	//	++data;
	//	//}

	//	for (int i = 0; i < area; i += 5)
	//	{
	//		float val1 = 0;
	//		float val2 = 0;
	//		float val3 = 0;
	//		float val4 = 0;
	//		float val5 = 0;
	//		float* w = weight;

	//		for (int inCh = 0; inCh < inChannel; inCh += 8)
	//		{
	//			val1 += *(d) * *(w)+ *(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
	//				*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

	//			d += 8;
	//			w += 8;
	//		}
	//		w = weight;

	//		for (int inCh = 0; inCh < inChannel; inCh += 8)
	//		{
	//			val2 += *(d) * *(w)+ *(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
	//				*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

	//			d += 8;
	//			w += 8;
	//		}
	//		w = weight;
	//		for (int inCh = 0; inCh < inChannel; inCh += 8)
	//		{
	//			val3 += *(d) * *(w)+ *(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
	//				*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

	//			d += 8;
	//			w += 8;
	//		}
	//		w = weight;
	//		for (int inCh = 0; inCh < inChannel; inCh += 8)
	//		{
	//			val4 += *(d) * *(w)+ *(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
	//				*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

	//			d += 8;
	//			w += 8;
	//		}
	//		w = weight;
	//		for (int inCh = 0; inCh < inChannel; inCh += 8)
	//		{
	//			val5 += *(d) * *(w)+ *(d + 1) * *(w + 1) + *(d + 2) * *(w + 2) + *(d + 3) * *(w + 3) +
	//				*(d + 4) * *(w + 4) + *(d + 5) * *(w + 5) + *(d + 6) * *(w + 6) + *(d + 7) * *(w + 7);

	//			d += 8;
	//			w += 8;
	//		}
	//		float b = *bias;
	//		float val[5] = { val1+b,val2 + b,val3 + b,val4 + b,val5 + b };
	//		memset(data, val, sizeof(float) * 5);
	//		//*(data) = val1 + b;
	//		//*(data+1) = val2 + b;
	//		//*(data+2) = val3 + b;
	//		//*(data+3) = val4 + b;
	//		//*(data+4) = val5 + b;

	//		data += 5;
	//		//*data = val + *bias;
	//		//++data;
	//	}



	//	weight += inChannel;
	//	++bias;
	//}



	Transpose(tensor);


	//float* o = data;
	//for (int outCh = 0; outCh < outChannel; ++outCh)
	//{
	//	float* d = o;
	//	float* v = tempTensor;

	//	float weightVal = *weight;
	//	for (int inCh = 0; inCh < inChannel; ++inCh)
	//	{
	//		o = d;
	//		float weightVal = *weight;

	//		for (int i = 0; i < area; ++i)
	//		{
	//			(*o) += *v * weightVal;
	//			++o;
	//			++v;
	//		}
	//		++weight;
	//	}
	//	++bias;
	//}






	//float* o = data;
	//for (int outCh = 0; outCh < outChannel; ++outCh)
	//{
	//	float* d = o;
	//	float* v = tempTensor;

	//	float weightVal = *weight;
	//	for (int i = 0; i < area; ++i)
	//	{
	//		(*o) += *v * weightVal + *bias;
	//		++o;
	//		++v;
	//	}
	//	for (int inCh = 1; inCh < inChannel; ++inCh)
	//	{
	//		o = d;
	//		float weightVal = *weight;

	//		for (int i = 0; i < area; ++i)
	//		{
	//			(*o) += *v * weightVal;
	//			++o;
	//			++v;
	//		}
	//		++weight;
	//	}
	//	++bias;
	//}

	tensor->channel = outChannel;
}















//void Mult(Tensor* tensor, Tensor* _tensor);
//void Equal(Tensor* tensor, Tensor* _tensor);
//void _Transpose(Tensor* tensor);
//void Transpose(Tensor* tensor);







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







//std::vector<topk> TopK(Tensor* tensor, int k);
//std::vector<Detection> Postprocessing(Tensor* offset, Tensor* size, Tensor* keypoint);

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
//	//tempTensor.data = new float[tensorDim * tensorDim * 64];
//	tempTensor.data = (float*)malloc(sizeof(float) * tensorDim * tensorDim * 64);
//
//	CopyTensor(&tempTensor, keypoint);
//	_MaxPool(&tempTensor, 3, 1, 1);
//
//	Equal(&tempTensor, keypoint);
//
//	Mult(keypoint, &tempTensor);
//	Transpose(keypoint);
//	free(tempTensor.data);
//
//	std::vector<topk> topkOutput = TopK(keypoint, k);
//
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
