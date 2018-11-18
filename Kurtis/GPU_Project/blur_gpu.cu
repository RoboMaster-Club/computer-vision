#include "myBlur_gpu.h"

int padWidth;
int padHeight;

__global__ void myBlur_gpu_kernal(InputArray src, OutputArray dst, Size ksize) {
	//get the index for a given block and thread, (blockDim = threads per block)
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < ksize.width && y < ksize.height) return;

	x += padWidth;
	y += padHeight;
	uchar3 sum = 0;

	//calculate the summation of the surounding pixels
	for (int i = x-padWidth; i <= x+padWidth; i++) {
		for (int j = y-padHeight; j <= y+padHeight; j++) {
				 sum += src(j, i);
		}
	}

	//set pixel as the average of the surounding pixels
	dst(y, x) = sum / (ksize.width * ksize.height);
}

void myBlur_gpu(InputArray src, OutputArray dst, Size ksize) {
	if (ksize.width == 0 || ksize.height == 0) return;

	//divide the array into blocks
	const int threadsPerBlock = 32;
	const dim3 gridSize(ceil((float) src.cols / threadsPerBlock), ceil((float) src.rows / threadsPerBlock), 1);
  	const dim3 blockSize(threadsPerBlock, threadsPerBlock, 1);

  	//allocate space for dst
  	dst.create(src.size(), src.type());

  	//pad image
  	padWidth = ceil((ksize.width - 1) / 2.0);
  	padHeight = ceil((ksize.height - 1) / 2.0);

  	InputArray _src();
  	_src.create(src.size(), src.type());
  	cv::copyMakeBorder(src, _src, padHeight, padHeight, padWidth, padWidth, BORDER_REPLICATE);

  	//apply average blur
	myBlur_gpu_kernal<<<gridSize, blockSize>>>(_src, dst, ksize);
}