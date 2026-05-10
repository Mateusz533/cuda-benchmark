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

	/* ==================================================================================================== */

	template<typename PixelType, typename Operation, typename... Args>
	__global__ void unaryOperation(DataAccessor<const PixelType> input, DataAccessor<PixelType> output, Size size, Args... args) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < size.width && y < size.height) {
			output.at(x, y) = Operation{}(input.get(x, y), args...);
		}
	}

	template<typename PixelType, typename Operation, typename... Args>
	__global__ void binaryOperation(DataAccessor<const PixelType> inputLeft, DataAccessor<const PixelType> inputRight, DataAccessor<PixelType> output, Size size, Args... args) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < size.width && y < size.height) {
			output.at(x, y) = Operation{}(inputLeft.get(x, y), inputRight.get(x, y), args...);
		}
	}

	template<typename PixelType, typename Operation, typename... Args>
	__global__ void nonlinearInputUnaryOperation(DataAccessor<const PixelType> input, DataAccessor<PixelType> output, Size size, Args... args) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < size.width && y < size.height) {
			output.at(x, y) = Operation{}(input, x, y, size, args...);
		}
	}

	template<typename PixelType, typename Operation, typename... Args>
	__global__ void nonlinearOutputUnaryOperation(DataAccessor<const PixelType> input, DataAccessor<PixelType> output, Size size, Args... args) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < size.width && y < size.height) {
			Operation{}(output, input.get(x, y), x, y, size, args...);
		}
	}

	template<typename PixelType, typename Operation, typename... Args>
	__global__ void nonlinearUnaryOperation(DataAccessor<const PixelType> input, DataAccessor<PixelType> output, Size size, Args... args) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(x < size.width && y < size.height) {
			Operation{}(input, output, x, y, size, args...);
		}
	}
}
