//#include "utils.h"
//#include "convolution_.h"
////#include "convolution_latency.h"
//
//#include "opencv2/opencv.hpp"
//
//
//std::vector<Detection> Inference();
//
//
//
//
//
//
//
//
//
//std::chrono::system_clock::time_point startTime;
//std::chrono::system_clock::time_point endTime;
//std::chrono::microseconds milli;
//std::chrono::microseconds total;
//
//
//std::chrono::microseconds dconv;
//std::chrono::microseconds pconv;
//std::chrono::microseconds nconv;
//std::chrono::microseconds resizeOps;
//std::chrono::microseconds addOps;
//std::chrono::microseconds maxpoolOps;
//std::chrono::microseconds concatOps;
//std::chrono::microseconds paddingOps;
//std::chrono::microseconds memcpyOps;
//std::chrono::microseconds transposeOps;
//
//
//
//
//
//std::unordered_map<std::string, Layer> layersMap;
//std::string weightsName = "D:/Project/SimpleConvolution/SimpleConvolution/detection_test.w";
//
//Tensor x;
//Tensor shortcutTensor;
//Tensor s4Tensor;
//Tensor s8Tensor;
//Tensor s16Tensor;
//
//Tensor offsetTensor;
//Tensor sizeTensor;
//Tensor keypointTensor;
//
//
//
//
//std::vector<Detection> Inference()
//{
//	int layerId = 1;
//	std::string layerIndex = std::to_string(layerId);
//
//
//	// conv1
//	nconv += _Convolution2D_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// blaze block 1 - single
//	CopyTensor(&shortcutTensor, &x);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//
//
//	// blaze block 2 - single
//	CopyTensor(&shortcutTensor, &x);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//
//	// blaze block 3 - single
//	CopyTensor(&shortcutTensor, &x);
//	maxpoolOps += _MaxPool(&shortcutTensor, 2, 2, 0);
//	concatOps += _ZeroConcat(&shortcutTensor);
//	dconv += _Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//	CopyTensor(&s4Tensor, &x);
//
//	// blaze block 4 - single
//	CopyTensor(&shortcutTensor, &x);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//
//	// blaze block 5 - single
//	CopyTensor(&shortcutTensor, &x);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//
//	// blaze block 6 - double
//	CopyTensor(&shortcutTensor, &x);
//	maxpoolOps += _MaxPool(&shortcutTensor, 2, 2, 0);
//	concatOps += _ZeroConcat(&shortcutTensor);
//	dconv += _Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//	CopyTensor(&s8Tensor, &x);
//
//	// blaze block 7 - double
//	CopyTensor(&shortcutTensor, &x);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//
//	// blaze block 8 - double
//	CopyTensor(&shortcutTensor, &x);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//
//	// blaze block 9 - double
//	CopyTensor(&shortcutTensor, &x);
//	maxpoolOps += _MaxPool(&shortcutTensor, 2, 2, 0);
//	dconv += _Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//	CopyTensor(&s16Tensor, &x);
//
//	// blaze block 10 - double
//	CopyTensor(&shortcutTensor, &x);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//
//	// blaze block 11 - double
//	CopyTensor(&shortcutTensor, &x);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&shortcutTensor, &x);
//
//
//	// fpn - feature map stride 4 - from blaze block 3
//	pconv += _Convolution2D_Pointwise_k1_s1(&s4Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// fpn - feature map stride 8 - from blaze block 6
//	pconv += _Convolution2D_Pointwise_k1_s1(&s8Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// fpn - feature map stride 16 - from blaze block 9
//	pconv += _Convolution2D_Pointwise_k1_s1(&s16Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// fpn - backbone
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	addOps += _Add(&s16Tensor, &x);
//
//	// fpn - stride 16
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	resizeOps += _Resize(&x, 2.0);
//	addOps += _Add(&s8Tensor, &x);
//
//	// fpn - stride 8
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	resizeOps += _Resize(&x, 2.0);
//	addOps += _Add(&s4Tensor, &x);
//
//	// fpn - stride 4
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	CopyTensor(&offsetTensor, &x, x.height, x.width, 2);
//
//	// head - offset block 1
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// head - offset block 2
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// head - offset block 3
//	pconv += _Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId); // offset out 
//
//	CopyTensor(&sizeTensor, &x, x.height, x.width, 2);
//
//	// head - size block 1
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// head - size block 2
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// head - size block 3
//	pconv += _Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId); // size out
//
//	CopyTensor(&keypointTensor, &x, x.height, x.width, 2);
//
//	// head - keypoint block 1
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// head - keypoint block 2
//	dconv += _Convolution2D_Depthwise_k3_s1_im2col(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//	pconv += _Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId);
//
//	// head - keypoint block 3
//	pconv += _Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//		layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//	layerIndex = std::to_string(++layerId); // keypoint out
//
//
//	std::vector<Detection> tempt;
//	tempt = Postprocessing(&offsetTensor, &sizeTensor, &keypointTensor);
//
//	return tempt;
//}
//
//
//
//
//
//
//
//
//
//
//
//
//
//int main()
//{
//	//std::chrono::system_clock::time_point startTime;
//	//std::chrono::system_clock::time_point endTime;
//	//std::chrono::microseconds milli;
//	//std::chrono::microseconds total;
//	//
//
//	//std::chrono::microseconds dconv;
//	//std::chrono::microseconds pconv;
//	//std::chrono::microseconds nconv;
//	//std::chrono::microseconds resizeOps;
//	//std::chrono::microseconds addOps;
//	//std::chrono::microseconds maxpoolOps;
//	//std::chrono::microseconds concatOps;
//	//std::chrono::microseconds paddingOps;
//	//std::chrono::microseconds memcpyOps;
//	//std::chrono::microseconds transposeOps;
//
//
//	total = total.zero();
//
//	dconv = dconv.zero();
//	pconv = pconv.zero();
//	nconv = nconv.zero();
//	resizeOps = resizeOps.zero();
//	addOps = addOps.zero();
//	maxpoolOps = maxpoolOps.zero();
//	concatOps = concatOps.zero();
//	paddingOps = paddingOps.zero();
//	memcpyOps = memcpyOps.zero();
//	transposeOps = transposeOps.zero();
//
//
//
//	//std::unordered_map<std::string, Layer> layersMap;
//
//	//std::string weightsName = "E:/vscode/Torch/MultiNet_OD_custom/src/KHI/utils/detection_test.w";
//	ReadWeights_debug(weightsName, layersMap);
//
//	for (std::pair<std::string, Layer> elem : layersMap)
//	{
//		std::cout << elem.first << " : " << "weightSize(" << elem.second.weightSize << "), inChannel(" << elem.second.inChannel << "), outChannel(" << elem.second.outChannel
//			<< "), kernel(" << elem.second.kernel << "), stride(" << elem.second.stride << "), padding(" << elem.second.padding << "), group(" << elem.second.group << ")" << std::endl;
//	}
//
//
//	int inputSize = 160;
//
//	int backboneTensorSize = inputSize * inputSize * 96 * 9;
//	int headTensorSize = inputSize * inputSize * 4 * 9;
//
//	x.width = inputSize;
//	x.height = inputSize;
//	x.channel = 3;
//	x.data = new float[backboneTensorSize];
//	for (int i = 0; i < backboneTensorSize; ++i)
//	{
//		x.data[i] = 1;
//	}
//
//	shortcutTensor.data = new float[backboneTensorSize];
//	s4Tensor.data = new float[backboneTensorSize];
//	s8Tensor.data = new float[backboneTensorSize];
//	s16Tensor.data = new float[backboneTensorSize];
//
//	offsetTensor.data = new float[headTensorSize];
//	sizeTensor.data = new float[headTensorSize];
//	keypointTensor.data = new float[headTensorSize];
//
//
//
//	//cv::VideoCapture capture("E:/carvi_dataset/TL/iphone/20210317D_TL.mp4");
//	//cv::Mat frame;
//
//	//while (!capture.isOpened())
//	//{
//	//	//capture >> frame;
//	//	capture.read(frame);
//	//	if (frame.empty())
//	//	{
//	//		printf("empty image");
//	//		//return 0;
//	//		break;
//	//	}
//	//	cv::Mat temp;
//	//	frame.copyTo(temp);
//	//	cv::resize(temp, temp, cv::Size(160, 160));
//	//	cv::cvtColor(temp, temp, cv::COLOR_BGR2RGB);
//	//	temp.convertTo(temp, CV_32FC3, 1.f / 127.5f, -1.0f);
//
//	//	memcpy(x.data, temp.data, sizeof(float) * temp.size().height * temp.size().width * temp.channels());
//	//	x.width = inputSize;
//	//	x.height = inputSize;
//	//	x.channel = temp.channels();
//
//
//	//	Inference();
//
//
//	//	cv::imshow("1", frame);
//	//	cv::waitKey(1);
//	//}
//
//
//
//
//
//	for (int i = 0; i < 100; ++i)
//	{
//		int layerId = 1;
//		std::string layerIndex = std::to_string(layerId);
//		x.width = inputSize;
//		x.height = inputSize;
//		x.channel = 3;
//		for (int i = 0; i < backboneTensorSize; ++i)
//		{
//			x.data[i] = 1;
//		}
//
//		// detection model
//		startTime = std::chrono::system_clock::now();
//
//		// conv1
//		nconv += _Convolution2D_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		_Relu(&x);
//		layerIndex = std::to_string(++layerId);
//
//		// blaze block 1 - single
//		CopyTensor(&shortcutTensor, &x);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//
//
//		// blaze block 2 - single
//		CopyTensor(&shortcutTensor, &x);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//
//		// blaze block 3 - single
//		CopyTensor(&shortcutTensor, &x);
//		maxpoolOps += _MaxPool(&shortcutTensor, 2, 2, 0);
//		concatOps += _ZeroConcat(&shortcutTensor);
//		dconv += _Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//		CopyTensor(&s4Tensor, &x);
//
//		// blaze block 4 - single
//		CopyTensor(&shortcutTensor, &x);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//
//		// blaze block 5 - single
//		CopyTensor(&shortcutTensor, &x);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//
//		// blaze block 6 - double
//		CopyTensor(&shortcutTensor, &x);
//		maxpoolOps += _MaxPool(&shortcutTensor, 2, 2, 0);
//		concatOps += _ZeroConcat(&shortcutTensor);
//		dconv += _Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//		CopyTensor(&s8Tensor, &x);
//
//		// blaze block 7 - double
//		CopyTensor(&shortcutTensor, &x);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//
//		// blaze block 8 - double
//		CopyTensor(&shortcutTensor, &x);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//
//		// blaze block 9 - double
//		CopyTensor(&shortcutTensor, &x);
//		maxpoolOps += _MaxPool(&shortcutTensor, 2, 2, 0);
//		dconv += _Convolution2D_Depthwise_k3_s2(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//		CopyTensor(&s16Tensor, &x);
//
//		// blaze block 10 - double
//		CopyTensor(&shortcutTensor, &x);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//
//		// blaze block 11 - double
//		CopyTensor(&shortcutTensor, &x);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&shortcutTensor, &x);
//		_Relu(&x);
//
//
//		// fpn - feature map stride 4 - from blaze block 3
//		pconv += _Convolution2D_Pointwise_k1_s1(&s4Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//
//		// fpn - feature map stride 8 - from blaze block 6
//		pconv += _Convolution2D_Pointwise_k1_s1(&s8Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//
//		// fpn - feature map stride 16 - from blaze block 9
//		pconv += _Convolution2D_Pointwise_k1_s1(&s16Tensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//
//		// fpn - backbone
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		addOps += _Add(&s16Tensor, &x);
//		_Relu(&x);
//
//		// fpn - stride 16
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		resizeOps += _Resize(&x, 2.0);
//		addOps += _Add(&s8Tensor, &x);
//		_Relu(&x);
//
//		// fpn - stride 8
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		resizeOps += _Resize(&x, 2.0);
//		addOps += _Add(&s4Tensor, &x);
//		_Relu(&x);
//
//		// fpn - stride 4
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&x, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//
//		CopyTensor(&offsetTensor, &x, x.height, x.width, 2);
//
//		// head - offset block 1
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		_Relu(&x);
//		layerIndex = std::to_string(++layerId);
//
//		// head - offset block 2
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		_Relu(&x);
//		layerIndex = std::to_string(++layerId);
//
//		// head - offset block 3
//		pconv += _Convolution2D_Pointwise_k1_s1(&offsetTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId); // offset out 
//
//		CopyTensor(&sizeTensor, &x, x.height, x.width, 2);
//
//		// head - size block 1
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		_Relu(&x);
//		layerIndex = std::to_string(++layerId);
//
//		// head - size block 2
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		_Relu(&x);
//		layerIndex = std::to_string(++layerId);
//
//		// head - size block 3
//		pconv += _Convolution2D_Pointwise_k1_s1(&sizeTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId); // size out
//
//		CopyTensor(&keypointTensor, &x, x.height, x.width, 2);
//
//		// head - keypoint block 1
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		_Relu(&x);
//		layerIndex = std::to_string(++layerId);
//
//		// head - keypoint block 2
//		dconv += _Convolution2D_Depthwise_k3_s1_im2col(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId);
//		pconv += _Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		_Relu(&x);
//		layerIndex = std::to_string(++layerId);
//
//		// head - keypoint block 3
//		pconv += _Convolution2D_Pointwise_k1_s1(&keypointTensor, layersMap[layerIndex].weights.data(), layersMap[layerIndex].bias.data(),
//			layersMap[layerIndex].inChannel, layersMap[layerIndex].outChannel, layersMap[layerIndex].kernel, layersMap[layerIndex].stride, layersMap[layerIndex].padding);
//		layerIndex = std::to_string(++layerId); // keypoint out
//
//
//		std::vector<Detection> temp;
//		temp = Postprocessing(&offsetTensor, &sizeTensor, &keypointTensor);
//
//
//		endTime = std::chrono::system_clock::now();
//		milli = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//		total += milli;
//	}
//
//	std::cout << "detection average : " << total.count() / 100 << " us ... " << total.count() / 100 / 1000 << " ms" << std::endl;
//
//	std::cout << "detection nconv average : " << nconv.count() / 100 << " us ... " << nconv.count() / 100 / 1000 << " ms" << std::endl;
//	std::cout << "detection dconv average : " << dconv.count() / 100 << " us ... " << dconv.count() / 100 / 1000 << " ms" << std::endl;
//	std::cout << "detection pconv average : " << pconv.count() / 100 << " us ... " << pconv.count() / 100 / 1000 << " ms" << std::endl;
//
//	std::cout << "detection resize ops average : " << resizeOps.count() / 100 << " us ... " << resizeOps.count() / 100 / 1000 << " ms" << std::endl;
//	std::cout << "detection add ops average : " << addOps.count() / 100 << " us ... " << addOps.count() / 100 / 1000 << " ms" << std::endl;
//	std::cout << "detection maxpool ops average : " << maxpoolOps.count() / 100 << " us ... " << maxpoolOps.count() / 100 / 1000 << " ms" << std::endl;
//	std::cout << "detection concat ops average : " << concatOps.count() / 100 << " us ... " << concatOps.count() / 100 / 1000 << " ms" << std::endl;
//	std::cout << "detection padding ops average : " << paddingOps.count() / 100 << " us ... " << paddingOps.count() / 100 / 1000 << " ms" << std::endl;
//
//	std::cout << "detection transpose ops average : " << transposeOps.count() / 100 << " us ... " << transposeOps.count() / 100 / 1000 << " ms" << std::endl;
//
//
//	total = total.zero();
//
//
//	Tensor clsTensor;
//	clsTensor.height = 16;
//	clsTensor.width = 16;
//	clsTensor.channel = 3;
//	clsTensor.data = new float[sizeof(float) * 16 * 16 * 32];
//	// classification model
//	startTime = std::chrono::system_clock::now();
//
//	_Convolution2D_k3_s2(&clsTensor, layersMap["1"].weights.data(), layersMap["1"].bias.data(), 3, 32, 3, 2, 1);
//
//	_Convolution2D_Depthwise_k3_s2(&clsTensor, layersMap["2"].weights.data(), layersMap["2"].bias.data(), 32, 32, 3, 2, 1);
//	_Convolution2D_Pointwise_k1_s1(&clsTensor, layersMap["3"].weights.data(), layersMap["3"].bias.data(), 32, 64, 1, 1, 1);
//
//	_Convolution2D_Depthwise_k3_s2(&clsTensor, layersMap["4"].weights.data(), layersMap["4"].bias.data(), 64, 64, 3, 2, 1);
//	_Convolution2D_Pointwise_k1_s1(&clsTensor, layersMap["5"].weights.data(), layersMap["5"].bias.data(), 64, 32, 1, 1, 1);
//
//	_Convolution2D_Depthwise_k3_s2(&clsTensor, layersMap["6"].weights.data(), layersMap["6"].bias.data(), 32, 32, 3, 2, 1);
//	_Convolution2D_Pointwise_k1_s1(&clsTensor, layersMap["7"].weights.data(), layersMap["7"].bias.data(), 32, 16, 1, 1, 1);
//
//	_Convolution2D_Pointwise_k1_s1(&clsTensor, layersMap["8"].weights.data(), layersMap["8"].bias.data(), 16, 7, 1, 1, 1);
//
//	_Softmax(&clsTensor);
//
//	endTime = std::chrono::system_clock::now();
//	milli = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
//	std::cout << "model : " << milli.count() << " us" << std::endl;
//
//	delete[] clsTensor.data;
//
//
//
//
//
//	delete[] x.data;
//	delete[] shortcutTensor.data;
//	delete[] s4Tensor.data;
//	delete[] s8Tensor.data;
//	delete[] s16Tensor.data;
//
//	delete[] offsetTensor.data;
//	delete[] sizeTensor.data;
//	delete[] keypointTensor.data;
//
//	return 0;
//}
