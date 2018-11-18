#ifndef _BLUR_GPU_H_
#define _BLUR_GPU_H_

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

/*
	@breif blur an image with a normalized kernal

	@param 	src 	the input image
	@param 	dst 	the output image
	@param 	ksize 	the size (width / height) of the kernal
*/
void blur_gpu(InputArray src, OutputArray dst, Size ksize);

#endif