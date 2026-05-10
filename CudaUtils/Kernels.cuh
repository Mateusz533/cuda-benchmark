#pragma once
//
#include "DeviceFunctions.cuh"

namespace CudaUtils::Kernels
{
	// Example CUDA kernel to image color inversion
	__global__ void processImage(const uchar3* input, uchar3* output, int width, int height, int inPitch, int outPitch) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < width && y < height) {
			const uchar3 inputPixel = get(input, inPitch, x, y);
			uchar3& outputPixel = at(output, outPitch, x, y);
			// USER CODE BEGIN
			outputPixel = CudaUtils::processImage(inputPixel);
			// USER CODE END
		}
	}

	/* ==================================================================================================== */

	template<typename PixelType>
	__global__ void invertColor(DataAccessor<const PixelType> input, DataAccessor<PixelType> output, Size size) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < size.width && y < size.height) {
			const PixelType inputPixel = input.get(x, y);
			PixelType& outputPixel = output.at(x, y);
			outputPixel = CudaUtils::invertColor(inputPixel);
		}
	}

	template<typename PixelType, typename Operation>
	__global__ void unaryOperation(DataAccessor<const PixelType> input, DataAccessor<PixelType> output, Size size) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < size.width && y < size.height) {
			const PixelType inputPixel = input.get(x, y);
			PixelType& outputPixel = output.at(x, y);
			outputPixel = Operation{}(inputPixel);
		}
	}

	template<typename PixelType, typename Operation>
	__global__ void binaryOperation(DataAccessor<const PixelType> inputLeft, DataAccessor<const PixelType> inputRight, DataAccessor<PixelType> output, Size size) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < size.width && y < size.height) {
			const PixelType inputLeftPixel = inputLeft.get(x, y);
			const PixelType inputRightPixel = inputRight.get(x, y);
			PixelType& outputPixel = output.at(x, y);
			outputPixel = Operation{}(inputLeftPixel, inputRightPixel);
		}
	}
}
