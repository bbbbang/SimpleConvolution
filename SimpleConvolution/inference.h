
#include "utils.h"
#include "convolution_test.h"





std::vector<Detection> Inference();

std::vector<Detection> Inference()
{
	int layerId = 1;
	std::string layerIndex = std::to_string(layerId);

	// conv1
	_Convolution2D_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// blaze block 1 - single
	CopyTensor(&shortcutTensor, &x);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);


	// blaze block 2 - single
	CopyTensor(&shortcutTensor, &x);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);

	// blaze block 3 - single
	CopyTensor(&shortcutTensor, &x);
	_MaxPool(&shortcutTensor, 2, 2, 0);
	_ZeroConcat(&shortcutTensor);
	_Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);
	CopyTensor(&s4Tensor, &x);

	// blaze block 4 - single
	CopyTensor(&shortcutTensor, &x);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);

	// blaze block 5 - single
	CopyTensor(&shortcutTensor, &x);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);

	// blaze block 6 - double
	CopyTensor(&shortcutTensor, &x);
	_MaxPool(&shortcutTensor, 2, 2, 0);
	_ZeroConcat(&shortcutTensor);
	_Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);
	CopyTensor(&s8Tensor, &x);

	// blaze block 7 - double
	CopyTensor(&shortcutTensor, &x);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);

	// blaze block 8 - double
	CopyTensor(&shortcutTensor, &x);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);

	// blaze block 9 - double
	CopyTensor(&shortcutTensor, &x);
	_MaxPool(&shortcutTensor, 2, 2, 0);
	_Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);
	CopyTensor(&s16Tensor, &x);

	// blaze block 10 - double
	CopyTensor(&shortcutTensor, &x);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);

	// blaze block 11 - double
	CopyTensor(&shortcutTensor, &x);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&shortcutTensor, &x);


	// fpn - feature map stride 4 - from blaze block 3
	_Convolution2D_Pointwise_k1_s1(&s4Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// fpn - feature map stride 8 - from blaze block 6
	_Convolution2D_Pointwise_k1_s1(&s8Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// fpn - feature map stride 16 - from blaze block 9
	_Convolution2D_Pointwise_k1_s1(&s16Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// fpn - backbone
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Add(&s16Tensor, &x);

	// fpn - stride 16
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Resize(&x, 2.0);
	_Add(&s8Tensor, &x);

	// fpn - stride 8
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Resize(&x, 2.0);
	_Add(&s4Tensor, &x);

	// fpn - stride 4
	_Convolution2D_Depthwise_k3_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	CopyTensor(&offsetTensor, &x, x.height, x.width, 2);

	// head - offset block 1
	_Convolution2D_Depthwise_k3_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// head - offset block 2
	_Convolution2D_Depthwise_k3_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// head - offset block 3
	_Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId); // offset out 

	CopyTensor(&sizeTensor, &x, x.height, x.width, 2);

	// head - size block 1
	_Convolution2D_Depthwise_k3_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// head - size block 2
	_Convolution2D_Depthwise_k3_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// head - size block 3
	_Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId); // size out

	CopyTensor(&keypointTensor, &x, x.height, x.width, 2);

	// head - keypoint block 1
	_Convolution2D_Depthwise_k3_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// head - keypoint block 2
	_Convolution2D_Depthwise_k3_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);
	_Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId);

	// head - keypoint block 3
	_Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
	layerIndex = std::to_string(++layerId); // keypoint out


	std::vector<Detection> tempt;
	tempt = Postprocessing(&offsetTensor, &sizeTensor, &keypointTensor);

	return tempt;
}