#include "convolution.h"

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

struct Layer
{
	//int width;
	//int height;
	int inChannel;
	int outChannel;
	int kernel;
	int stride;
	int padding;
	int group;

	int weightSize;

	std::vector<float> weights;
	std::vector<float> bias;

	//convolution conv;
};

void ReadWeights_debug(std::string fileName, std::unordered_map<std::string, Layer>& map)
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


int main()
{
	std::chrono::system_clock::time_point startTime;
	std::chrono::system_clock::time_point endTime;
	std::chrono::microseconds milli;
	std::chrono::microseconds total;
	total = total.zero();

	std::unordered_map<std::string, Layer> layersMap;

	std::string weightsName = "./detection_test.w";
	ReadWeights_debug(weightsName, layersMap);

	for (std::pair<std::string, Layer> elem : layersMap)
	{
		std::cout << elem.first << " : " << "weightSize(" << elem.second.weightSize << "), inChannel(" << elem.second.inChannel << "), outChannel(" << elem.second.outChannel
			<< "), kernel(" << elem.second.kernel << "), stride(" << elem.second.stride << "), padding(" << elem.second.padding << "), group(" << elem.second.group << ")" << std::endl;
	}

	int inputSize = 160;

	float* inputData = new float[inputSize * inputSize * 96];
	float* outputData = new float[inputSize * inputSize * 96];

	for (int i = 0; i < inputSize * inputSize * 96; ++i)
	{
		inputData[i] = 1;
		outputData[i] = 0;
	}


	std::chrono::microseconds dconv;
	std::chrono::microseconds pconv;
	std::chrono::microseconds nconv;

	dconv = dconv.zero();
	pconv = pconv.zero();
	nconv = nconv.zero();



	// temp
	//std::vector<float> weight;
	//std::vector<float> bias;

	//for (int i = 0; i < 8000; ++i)
	//{
	//	weight.push_back(1.1f);
	//	bias.push_back(1.1f);
	//}

	//for (int i = 1; i < 61; ++i)
	//{
	//	std::string id = std::to_string(i);
	//	layersMap[id] = Layer{ 48,48,3,1,1,1,48 * 48 * 3 * 3,weight,bias };
	//}



	for (int i = 0; i < 100; ++i)
	{
		int layerId = 1;
		std::string layerIndex = std::to_string(layerId);
		// detection model
		// 60
		startTime = std::chrono::system_clock::now();

		nconv += Convolution2D_k3_s2(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize, inputSize,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 2, inputSize / 2,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 2, inputSize / 2,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 2, inputSize / 2,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 2, inputSize / 2,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s2(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 2, inputSize / 2,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s2(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s2(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 16, inputSize / 16,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 8, inputSize / 8,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId); // offset out 

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId); // size out

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		dconv += Convolution2D_Depthwise_k3_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		pconv += Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(), inputSize / 4, inputSize / 4,
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId); // keypoint out

		endTime = std::chrono::system_clock::now();
		milli = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
		//std::cout << "model : " << milli.count() << " us ... " << milli.count() / 1000 << " ms" << std::endl;
		//std::cout << "layer num : " << layerId - 1 << std::endl << std::endl;
		total += milli;
	}

	std::cout << "detection average : " << total.count() / 100 << " us ... " << total.count() / 100 / 1000 << " ms" << std::endl;
	total = total.zero();
	std::cout << "detection nconv average : " << nconv.count() / 100 << " us ... " << nconv.count() / 100 / 1000 << " ms" << std::endl;
	std::cout << "detection dconv average : " << dconv.count() / 100 << " us ... " << dconv.count() / 100 / 1000 << " ms" << std::endl;
	std::cout << "detection pconv average : " << pconv.count() / 100 << " us ... " << pconv.count() / 100 / 1000 << " ms" << std::endl;


	// classification model
	startTime = std::chrono::system_clock::now();

	Convolution2D_k3_s2(inputData, outputData, layersMap["1"].weights.data(), layersMap["1"].bias.data(), 16, 16, 3, 32, 3, 2, 1);

	Convolution2D_Depthwise_k3_s2(inputData, outputData, layersMap["2"].weights.data(), layersMap["2"].bias.data(), 8, 8, 32, 32, 3, 2, 1);
	Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap["3"].weights.data(), layersMap["3"].bias.data(), 4, 4, 32, 64, 1, 1, 1);

	Convolution2D_Depthwise_k3_s2(inputData, outputData, layersMap["4"].weights.data(), layersMap["4"].bias.data(), 4, 4, 64, 64, 3, 2, 1);
	Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap["5"].weights.data(), layersMap["5"].bias.data(), 2, 2, 64, 32, 1, 1, 1);

	Convolution2D_Depthwise_k3_s2(inputData, outputData, layersMap["6"].weights.data(), layersMap["6"].bias.data(), 2, 2, 32, 32, 3, 2, 1);
	Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap["7"].weights.data(), layersMap["7"].bias.data(), 1, 1, 32, 16, 1, 1, 1);

	Convolution2D_Pointwise_k1_s1(inputData, outputData, layersMap["8"].weights.data(), layersMap["8"].bias.data(), 1, 1, 16, 7, 1, 1, 1);

	endTime = std::chrono::system_clock::now();
	milli = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
	std::cout << "model : " << milli.count() << " us" << std::endl;


	return 0;
}
