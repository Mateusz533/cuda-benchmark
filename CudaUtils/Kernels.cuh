#pragma once
//
#include "DeviceFunctions.cuh"

namespace CudaUtils
{
	// Example CUDA kernel to image color inversion
	__global__ void processImageKernel(const uchar3* input, uchar3* output, int width, int height, int inPitch, int outPitch) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < width && y < height) {
			const uchar3 inputPixel = get(input, inPitch, x, y);
			uchar3& outputPixel = at(output, outPitch, x, y);
			// USER CODE BEGIN
			outputPixel = processImage(inputPixel);
			// USER CODE END
		}
	}

	/* ==================================================================================================== */

	template<typename PixelType>
	__global__ void invertColorKernel(const PixelType* input, PixelType* output, int width, int height, int inPitch, int outPitch) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < width && y < height) {
			const PixelType inputPixel = get(input, inPitch, x, y);
			PixelType& outputPixel = at(output, outPitch, x, y);
			outputPixel = invertColor(inputPixel);
		}
	}

	template<typename PixelType, typename Operation>
	__global__ void unaryOperationKernel(const PixelType* input, PixelType* output, int width, int height, int inPitch, int outPitch) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < width && y < height) {
			const PixelType inputPixel = get(input, inPitch, x, y);
			PixelType& outputPixel = at(output, outPitch, x, y);
			outputPixel = Operation{}(inputPixel);
		}
	}

	template<typename PixelType, typename Operation>
	__global__ void binaryOperationKernel(const PixelType* inputLeft, const PixelType* inputRight, PixelType* output, int width, int height,
										  int inLeftPitch, int inRightPitch, int outPitch) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < width && y < height) {
			const PixelType inputLeftPixel = get(inputLeft, inLeftPitch, x, y);
			const PixelType inputRightPixel = get(inputRight, inRightPitch, x, y);
			PixelType& outputPixel = at(output, outPitch, x, y);
			outputPixel = Operation{}(inputLeftPixel, inputRightPixel);
		}
	}
}
