#include "conv.h"
#include <time.h>

int main()
{
	Tensor x;
	Tensor shortcutTensor;
	Tensor s4Tensor;
	Tensor s8Tensor;
	Tensor s16Tensor;

	Tensor offsetTensor;
	Tensor sizeTensor;
	Tensor keypointTensor;

	//Layer* layersMap = NULL;
	//layersMap = (Layer*)malloc(sizeof(Layer) * 60);
	Layer layersMap[60];

	//ReadWeights_binary("./tl_160.w", layersMap);
	ReadWeights_binary("E:/vscode/Torch/MultiNet_OD_custom/src/KHI/utils/tl_160.w", layersMap);

	for (int i = 0; i < 60; ++i)
	{
		printf("%d, %d\n", i, layersMap[i].weightSize);
	}


	int inputSize = 80;
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

	for (int i = 0; i < 200; ++i)
	{
		double start, end;
		start = (double)clock() / CLOCKS_PER_SEC;

		int layerId = 0;
		x.width = inputSize;
		x.height = inputSize;
		x.channel = 3;
		for (int i = 0; i < backboneTensorSize; ++i)
		{
			x.data[i] = 1;
		}

		// detection model
		// conv1
		_Convolution2D_k3_s2(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		_Relu(&x);
		++layerId;

		// blaze block 1 - single
		CopyTensor(&shortcutTensor, &x);
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 2 - single
		CopyTensor(&shortcutTensor, &x);
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 3 - single
		CopyTensor(&shortcutTensor, &x);
		_MaxPool(&shortcutTensor, 2, 2, 0);
		_ZeroConcat(&shortcutTensor);
		_Convolution2D_Depthwise_k3_s2(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);
		CopyTensor(&s4Tensor, &x);


		// blaze block 4 - single
		CopyTensor(&shortcutTensor, &x);
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 5 - single
		CopyTensor(&shortcutTensor, &x);
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 6 - double
		CopyTensor(&shortcutTensor, &x);
		_MaxPool(&shortcutTensor, 2, 2, 0);
		_ZeroConcat(&shortcutTensor);
		_Convolution2D_Depthwise_k3_s2(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);
		CopyTensor(&s8Tensor, &x);


		// blaze block 7 - double
		CopyTensor(&shortcutTensor, &x);
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 8 - double
		CopyTensor(&shortcutTensor, &x);
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 9 - double
		CopyTensor(&shortcutTensor, &x);
		_MaxPool(&shortcutTensor, 2, 2, 0);
		_Convolution2D_Depthwise_k3_s2(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);
		CopyTensor(&s16Tensor, &x);


		// blaze block 10 - double
		CopyTensor(&shortcutTensor, &x);
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);


		// blaze block 11 - double
		CopyTensor(&shortcutTensor, &x);
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&shortcutTensor, &x);
		_Relu(&x);

		// fpn - feature map stride 4 - from blaze block 3
		_Convolution2D_Pointwise_k1_s1(&s4Tensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;

		// fpn - feature map stride 8 - from blaze block 6
		_Convolution2D_Pointwise_k1_s1(&s8Tensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;

		// fpn - feature map stride 16 - from blaze block 9
		_Convolution2D_Pointwise_k1_s1(&s16Tensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;

		// fpn - backbone
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Add(&s16Tensor, &x);
		_Relu(&x);

		// fpn - stride 16
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Resize(&x, 2.0);
		_Add(&s8Tensor, &x);
		_Relu(&x);

		// fpn - stride 8
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Resize(&x, 2.0);
		_Add(&s4Tensor, &x);
		_Relu(&x);

		// fpn - stride 4
		_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;

		//CopyTensor(&offsetTensor, &x, x.height, x.width, x.channel);
		//CopyTensor(&sizeTensor, &x, x.height, x.width, x.channel);
		//CopyTensor(&keypointTensor, &x, x.height, x.width, x.channel);

		CopyTensor(&offsetTensor, &x);
		CopyTensor(&sizeTensor, &x);
		CopyTensor(&keypointTensor, &x);

		// head - offset block 1
		_Convolution2D_Depthwise_k3_s1(&offsetTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Relu(&offsetTensor);

		// head - offset block 2
		_Convolution2D_Depthwise_k3_s1(&offsetTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Relu(&offsetTensor);

		// head - offset block 3
		_Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId; // offset out 


		// head - size block 1
		_Convolution2D_Depthwise_k3_s1(&sizeTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Relu(&sizeTensor);

		// head - size block 2
		_Convolution2D_Depthwise_k3_s1(&sizeTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Relu(&sizeTensor);

		// head - size block 3
		_Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId; // size out


		// head - keypoint block 1
		_Convolution2D_Depthwise_k3_s1(&keypointTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Relu(&keypointTensor);

		// head - keypoint block 2
		_Convolution2D_Depthwise_k3_s1(&keypointTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId;
		_Relu(&keypointTensor);

		// head - keypoint block 3
		_Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerId].weights, layersMap[layerId].bias,
			layersMap[layerId].inChannel, layersMap[layerId].outChannel, layersMap[layerId].kernel, layersMap[layerId].stride, layersMap[layerId].padding);
		++layerId; // keypoint out


		end = (((double)clock()) / CLOCKS_PER_SEC);
		printf("time :%lf\n", (end - start));
	}





	free(x.data); x.data = NULL;
	free(shortcutTensor.data); shortcutTensor.data = NULL;
	free(s4Tensor.data); s4Tensor.data = NULL;
	free(s8Tensor.data); s8Tensor.data = NULL;
	free(s16Tensor.data); s16Tensor.data = NULL;
	free(offsetTensor.data); offsetTensor.data = NULL;
	free(sizeTensor.data); sizeTensor.data = NULL;
	free(keypointTensor.data); keypointTensor.data = NULL;



	//free(layersMap); layersMap = NULL;

	for (int i = 0; i < 60; ++i)
	{
		free(layersMap[i].weights);
		free(layersMap[i].bias);
	}

	return 0;
}
