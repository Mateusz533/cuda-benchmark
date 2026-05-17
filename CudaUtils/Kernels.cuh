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

	__global__ void reduceSumKernel(const float* __restrict__ partialSums, float* __restrict__ totalSum, int size) {
		constexpr int BLOCK_1D_SIZE = BLOCK_DIM * BLOCK_DIM;
		constexpr int WARPS_PER_BLOCK = BLOCK_1D_SIZE / WARP_SIZE;
		__shared__ float sharedSum[WARPS_PER_BLOCK];

		const int threadId = threadIdx.x;
		const int index = blockIdx.x * blockDim.x + threadId;
		const int warpId = threadId / WARP_SIZE;

		// Block-level reduction
		const float inputValue = (index < size) ? partialSums[index] : 0.0f;
		const float warpSum = warpReduceSum(inputValue);

		if(threadId % WARP_SIZE == 0) {
			sharedSum[warpId] = warpSum;
		}
		__syncthreads();

		if(warpId == 0) {
			const float warpSum = (threadId < WARPS_PER_BLOCK) ? sharedSum[threadId] : 0.0f;
			const float blockSum = warpReduceSum(warpSum);
			if(threadId == 0) {
				atomicAdd(totalSum, blockSum);
			}
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
