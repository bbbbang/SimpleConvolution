
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>




//// struct
typedef struct _Tensor
{
	int height;
	int width;
	int channel;

	float* data;
}Tensor;


typedef struct _Layer
{
	int inChannel;
	int outChannel;
	int kernel;
	int stride;
	int padding;
	int group;

	int weightSize;

	float* weights;
	float* bias;

	//convolution conv;
}Layer;





#define BUFFER_SIZE 256

void ReadWeights_debug(const char* fileName, Layer* map);
void ReadWeights_txt(const char* fileName, Layer* map);

void ReadWeights_debug(const char* fileName, Layer* map)
{
	std::ifstream weightFile(fileName, std::ios::binary);
	

	if (weightFile.is_open())
	{
		float data;

		int inChannel;
		int outChannel;
		int kernel;
		int stride;
		int padding;
		int group;

		std::vector<float> weights;
		std::vector<float> bias;
		int weightSize;

		int layersNum = 0;
		int weightNum = 0;
		int biasNum = 0;

		while (weightFile)
		{
			weightFile.read(reinterpret_cast<char*>(&data), sizeof(float));
			//std::cout << weight << std::endl;

			if ((int)data == 9000)
			{
				if (layersNum > 0)
				{
					map[std::to_string(layersNum)] = Layer{ inChannel, outChannel, kernel, stride, padding, group, weightSize, weights, bias };
				}
				weights.clear();
				bias.clear();
				weightNum = 0;
				biasNum = 0;
				layersNum += 1;

				weightFile.read(reinterpret_cast<char*>(&data), sizeof(float));
				inChannel = (int)data;
				weightFile.read(reinterpret_cast<char*>(&data), sizeof(float));
				outChannel = (int)data;
				weightFile.read(reinterpret_cast<char*>(&data), sizeof(float));
				kernel = (int)data;
				weightFile.read(reinterpret_cast<char*>(&data), sizeof(float));
				stride = (int)data;
				weightFile.read(reinterpret_cast<char*>(&data), sizeof(float));
				padding = (int)data;
				weightFile.read(reinterpret_cast<char*>(&data), sizeof(float));
				group = (int)data;

				if (group == 1)
				{
					weightSize = inChannel * outChannel * kernel * kernel;
				}
				else
				{
					weightSize = inChannel * 1 * kernel * kernel;
				}
			}
			else
			{
				if (weightNum < weightSize)
				{
					weights.push_back(data);
					weightNum += 1;
				}
				else if (weightNum >= weightSize && biasNum < outChannel)
				{
					bias.push_back(data);
					biasNum += 1;
				}
			}
		}
		map[std::to_string(layersNum)] = Layer{ inChannel, outChannel, kernel, stride, padding, group, weightSize, weights, bias };
		weightFile.close();
	}
	else
	{
		std::cout << "no file" << std::endl;
	}

}

void ReadWeights_txt(const char* fileName, Layer* map)
{
	FILE* fp = fopen(fileName, "r");

	char buffer[BUFFER_SIZE + 1];

	if (weightFile.is_open())
	{
		float data;

		int inChannel;
		int outChannel;
		int kernel;
		int stride;
		int padding;
		int group;

		std::vector<float> weights;
		std::vector<float> bias;
		int weightSize;

		int layersNum = 0;
		int weightNum = 0;
		int biasNum = 0;

		while (weightFile)
		{
			weightFile >> data;

			if ((int)data == 9000)
			{
				if (layersNum > 0)
				{
					map[std::to_string(layersNum)] = Layer{ inChannel, outChannel, kernel, stride, padding, group, weightSize, weights, bias };
				}
				weights.clear();
				bias.clear();
				weightNum = 0;
				biasNum = 0;
				layersNum += 1;

				weightFile >> data;
				inChannel = (int)data;
				weightFile >> data;
				outChannel = (int)data;
				weightFile >> data;
				kernel = (int)data;
				weightFile >> data;
				stride = (int)data;
				weightFile >> data;
				padding = (int)data;
				weightFile >> data;
				group = (int)data;

				if (group == 1)
				{
					weightSize = inChannel * outChannel * kernel * kernel;
				}
				else
				{
					weightSize = inChannel * 1 * kernel * kernel;
				}
			}
			else
			{
				if (weightNum < weightSize)
				{
					weights.push_back(data);
					weightNum += 1;
				}
				else if (weightNum >= weightSize && biasNum < outChannel)
				{
					bias.push_back(data);
					biasNum += 1;
				}
			}
		}
		map[std::to_string(layersNum)] = Layer{ inChannel, outChannel, kernel, stride, padding, group, weightSize, weights, bias };
		weightFile.close();
	}
	else
	{
		std::cout << "no file" << std::endl;
	}

}
















//// functions




void CopyTensor(Tensor* dst, Tensor* src);
void CopyTensor(Tensor* dst, Tensor* src, int height, int width, int channel);

void CopyTensor(Tensor* dst, Tensor* src)
{
	dst->height = src->height;
	dst->width = src->width;
	dst->channel = src->channel;
	memcpy(dst->data, src->data, sizeof(float) * dst->height * dst->width * dst->channel);
}

void CopyTensor(Tensor* dst, Tensor* src, int height, int width, int channel)
{
	dst->height = height;
	dst->width = width;
	dst->channel = channel;
	memcpy(dst->data, src->data, sizeof(float) * dst->height * dst->width * dst->channel);
}




