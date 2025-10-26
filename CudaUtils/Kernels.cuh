#pragma once
//
#include "DeviceFunctions.cuh"

namespace CudaUtils
{
	// Example CUDA kernel to image color inversion
	__global__ void invertColorKernel(const uchar3* input, uchar3* output, int width, int height, int pitch) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < width && y < height) {
			const uchar3& inputPixel = get(input, pitch, x, y);
			uchar3& outputPixel = at(output, pitch, x, y);
			// USER CODE BEGIN
			outputPixel = invertColor(inputPixel);
			// USER CODE END
		}
	}

	/* ==================================================================================================== */
}
