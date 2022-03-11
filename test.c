#include "convolution_c.h"
#include "utils_c.h"








int main()
{
	//std::unordered_map<std::string, Layer> layersMap;
	//std::string weightsName = "E:/vscode/Torch/MultiNet_OD_custom/src/KHI/utils/tl_160.txt";

	Tensor x;
	Tensor shortcutTensor;
	Tensor s4Tensor;
	Tensor s8Tensor;
	Tensor s16Tensor;

	Tensor offsetTensor;
	Tensor sizeTensor;
	Tensor keypointTensor;
	
	//std::unordered_map<std::string, Layer> layersMap;

	//std::string weightsName = "E:/vscode/Torch/MultiNet_OD_custom/src/KHI/utils/detection_test.w";
	//ReadWeights_debug(weightsName, layersMap);
	//ReadWeights_txt(weightsName, layersMap);

	//for (std::pair<std::string, Layer> elem : layersMap)
	//{
	//	std::cout << elem.first << " : " << "weightSize(" << elem.second.weightSize << "), inChannel(" << elem.second.inChannel << "), outChannel(" << elem.second.outChannel
	//		<< "), kernel(" << elem.second.kernel << "), stride(" << elem.second.stride << "), padding(" << elem.second.padding << "), group(" << elem.second.group << ")" << std::endl;
	//}


	Layer layersMap[60];

	int inputSize = 160;

	int backboneTensorSize = inputSize * inputSize * 96;
	int headTensorSize = inputSize * inputSize * 4;

	x.width = inputSize;
	x.height = inputSize;
	x.channel = 3;
	x.data = (float*)malloc(sizeof(float) * backboneTensorSize);

	for (int i = 0; i < backboneTensorSize; ++i)
	{
		x.data[i] = 1;
	}

	shortcutTensor.data = (float*)malloc(sizeof(float) * backboneTensorSize);
	s4Tensor.data = (float*)malloc(sizeof(float) * backboneTensorSize);
	s8Tensor.data = (float*)malloc(sizeof(float) * backboneTensorSize);
	s16Tensor.data = (float*)malloc(sizeof(float) * backboneTensorSize);

	offsetTensor.data = (float*)malloc(sizeof(float) * headTensorSize);
	sizeTensor.data = (float*)malloc(sizeof(float) * headTensorSize);
	keypointTensor.data = (float*)malloc(sizeof(float) * headTensorSize);

	for (int i = 0; i < 100; ++i)
	{
		int layerId = 1;
		std::string layerIndex = std::to_string(layerId);
		x.width = inputSize;
		x.height = inputSize;
		x.channel = 3;
		for (int i = 0; i < backboneTensorSize; ++i)
		{
			x.data[i] = 1;
		}

		// detection model
		startTime = std::chrono::system_clock::now();

		// conv1
		nconv += _Convolution2D_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		_Relu(&x);
		layerIndex = std::to_string(++layerId);

		// blaze block 1 - single
		CopyTensor(&shortcutTensor, &x);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 2 - single
		CopyTensor(&shortcutTensor, &x);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 3 - single
		CopyTensor(&shortcutTensor, &x);
		maxpoolOps += _MaxPool(&shortcutTensor, 2, 2, 0);
		concatOps += _ZeroConcat(&shortcutTensor);
		dconv += _Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);
		CopyTensor(&s4Tensor, &x);


		// blaze block 4 - single
		CopyTensor(&shortcutTensor, &x);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 5 - single
		CopyTensor(&shortcutTensor, &x);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 6 - double
		CopyTensor(&shortcutTensor, &x);
		maxpoolOps += _MaxPool(&shortcutTensor, 2, 2, 0);
		concatOps += _ZeroConcat(&shortcutTensor);
		dconv += _Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);
		CopyTensor(&s8Tensor, &x);


		// blaze block 7 - double
		CopyTensor(&shortcutTensor, &x);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 8 - double
		CopyTensor(&shortcutTensor, &x);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 9 - double
		CopyTensor(&shortcutTensor, &x);
		maxpoolOps += _MaxPool(&shortcutTensor, 2, 2, 0);
		dconv += _Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);
		CopyTensor(&s16Tensor, &x);


		// blaze block 10 - double
		CopyTensor(&shortcutTensor, &x);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 11 - double
		CopyTensor(&shortcutTensor, &x);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&shortcutTensor, &x);
		_Relu(&x);

		// fpn - feature map stride 4 - from blaze block 3
		pconv += _Convolution2D_Pointwise_k1_s1(&s4Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		// fpn - feature map stride 8 - from blaze block 6
		pconv += _Convolution2D_Pointwise_k1_s1(&s8Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		// fpn - feature map stride 16 - from blaze block 9
		pconv += _Convolution2D_Pointwise_k1_s1(&s16Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		// fpn - backbone
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		addOps += _Add(&s16Tensor, &x);
		_Relu(&x);

		// fpn - stride 16
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		resizeOps += _Resize(&x, 2.0);
		addOps += _Add(&s8Tensor, &x);
		_Relu(&x);

		// fpn - stride 8
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		resizeOps += _Resize(&x, 2.0);
		addOps += _Add(&s4Tensor, &x);
		_Relu(&x);

		// fpn - stride 4
		dconv += _Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);

		CopyTensor(&offsetTensor, &x, x.height, x.width, x.channel);
		CopyTensor(&sizeTensor, &x, x.height, x.width, x.channel);
		CopyTensor(&keypointTensor, &x, x.height, x.width, x.channel);

		// head - offset block 1
		dconv += _Convolution2D_Depthwise_k3_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		_Relu(&offsetTensor);

		// head - offset block 2
		dconv += _Convolution2D_Depthwise_k3_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		_Relu(&offsetTensor);

		// head - offset block 3
		pconv += _Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId); // offset out 

		
		// head - size block 1
		dconv += _Convolution2D_Depthwise_k3_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		_Relu(&sizeTensor);

		// head - size block 2
		dconv += _Convolution2D_Depthwise_k3_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		_Relu(&sizeTensor);

		// head - size block 3
		pconv += _Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId); // size out

		
		// head - keypoint block 1
		dconv += _Convolution2D_Depthwise_k3_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		_Relu(&keypointTensor);

		// head - keypoint block 2
		dconv += _Convolution2D_Depthwise_k3_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		pconv += _Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId);
		_Relu(&keypointTensor);

		// head - keypoint block 3
		pconv += _Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
		layerIndex = std::to_string(++layerId); // keypoint out


		endTime = std::chrono::system_clock::now();
		milli = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
		total += milli;
	}

	std::cout << "detection average : " << total.count() / 100 << " us ... " << total.count() / 100 / 1000 << " ms" << std::endl;
	total = total.zero();


	//Tensor clsTensor;
	//clsTensor.height = 16;
	//clsTensor.width = 16;
	//clsTensor.channel = 3;
	//clsTensor.data = new float[sizeof(float) * 16 * 16 * 32];
	//// classification model
	//startTime = std::chrono::system_clock::now();

	//_Convolution2D_k3_s2(&clsTensor, layersMap["1"].weights.data(), layersMap["1"].bias.data(), 3, 32, 3, 2, 1);

	//_Convolution2D_Depthwise_k3_s2(&clsTensor, layersMap["2"].weights.data(), layersMap["2"].bias.data(), 32, 32, 3, 2, 1);
	//_Convolution2D_Pointwise_k1_s1(&clsTensor, layersMap["3"].weights.data(), layersMap["3"].bias.data(), 32, 64, 1, 1, 1);

	//_Convolution2D_Depthwise_k3_s2(&clsTensor, layersMap["4"].weights.data(), layersMap["4"].bias.data(), 64, 64, 3, 2, 1);
	//_Convolution2D_Pointwise_k1_s1(&clsTensor, layersMap["5"].weights.data(), layersMap["5"].bias.data(), 64, 32, 1, 1, 1);

	//_Convolution2D_Depthwise_k3_s2(&clsTensor, layersMap["6"].weights.data(), layersMap["6"].bias.data(), 32, 32, 3, 2, 1);
	//_Convolution2D_Pointwise_k1_s1(&clsTensor, layersMap["7"].weights.data(), layersMap["7"].bias.data(), 32, 16, 1, 1, 1);

	//_Convolution2D_Pointwise_k1_s1(&clsTensor, layersMap["8"].weights.data(), layersMap["8"].bias.data(), 16, 7, 1, 1, 1);

	//_Softmax(&clsTensor);

	//endTime = std::chrono::system_clock::now();
	//milli = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
	//std::cout << "model : " << milli.count() << " us" << std::endl;

	//delete[] clsTensor.data;



	free(x.data); x.data = NULL;
	free(shortcutTensor.data); shortcutTensor.data = NULL;
	free(s4Tensor.data); s4Tensor.data = NULL;
	free(s8Tensor.data); s8Tensor.data = NULL;
	free(s16Tensor.data); s16Tensor.data = NULL;
	free(offsetTensor.data); offsetTensor.data = NULL;
	free(sizeTensor.data); sizeTensor.data = NULL;
	free(keypointTensor.data); keypointTensor.data = NULL;

	return 0;
}
