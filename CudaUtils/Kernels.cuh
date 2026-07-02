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

	__global__ void reduceSum(const float* __restrict__ partialSums, float* __restrict__ totalSum, int size) {
		const int threadId = threadIdx.x;
		const int index = blockIdx.x * blockDim.x + threadId;

		const float inputValue = (index < size) ? partialSums[index] : 0.0f;

		blockReduceSum(inputValue, threadId, [totalSum](float blockSum) {
			atomicAdd(totalSum, blockSum);
		});
	}

	template<int MASK_DIM = 1>
		requires(MASK_DIM > 0 && MASK_DIM % 2 == 1)
	__global__ void laplacianSquareSum(DataAccessor<const uchar> input, float* blockSums, Size size) {
		constexpr int HALO_WIDTH = (MASK_DIM > 1) ? (MASK_DIM - 1) / 2 : 1;
		constexpr int TILE_DIM = SQUARE_BLOCK_DIM + 2 * HALO_WIDTH;

		__shared__ int tile[TILE_DIM][TILE_DIM];
		TileView tileView{tile};

		const int unifiedThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;
		const int unifiedBlockIdx = blockIdx.y * gridDim.x + blockIdx.x;

		// Copy block and halo to shared memory
		constexpr int PIXELS_TO_COPY_BY_THREAD = (TILE_DIM * TILE_DIM + BLOCK_TOTAL_SIZE - 1) / (BLOCK_TOTAL_SIZE);

		for(int i = 0; i < PIXELS_TO_COPY_BY_THREAD; ++i) {
			const int tilePixelIdx = unifiedThreadIdx + i * BLOCK_TOTAL_SIZE;

			if(tilePixelIdx < TILE_DIM * TILE_DIM) {
				const int tilePixelX = tilePixelIdx % TILE_DIM;
				const int tilePixelY = tilePixelIdx / TILE_DIM;

				const int inputX = tilePixelX + blockIdx.x * blockDim.x - HALO_WIDTH;
				const int inputY = tilePixelY + blockIdx.y * blockDim.y - HALO_WIDTH;

				const bool inputInRange = isInRange(inputX, 0, size.width - 1) && isInRange(inputY, 0, size.height - 1);
				tileView.at(tilePixelX, tilePixelX) = inputInRange ? input.get(inputX, inputY) : 0;
			}
		}
		__syncthreads();

		// Compute Laplacian
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		const int tileX = threadIdx.x + HALO_WIDTH;
		const int tileY = threadIdx.y + HALO_WIDTH;

		int laplacianValue = 0;

		if(isInRange(x, HALO_WIDTH, size.width - 1 - HALO_WIDTH) && isInRange(y, HALO_WIDTH, size.height - 1 - HALO_WIDTH)) {
			if constexpr(MASK_DIM == 1) {
				laplacianValue = -4 * tileView.get(tileX, tileY);
				laplacianValue += tileView.get(tileX, tileY - 1);
				laplacianValue += tileView.get(tileX, tileY + 1);
				laplacianValue += tileView.get(tileX - 1, tileY);
				laplacianValue += tileView.get(tileX + 1, tileY);
			} else if constexpr(MASK_DIM == 3) {
				laplacianValue = -8 * tileView.get(tileX, tileY);
				laplacianValue += 2 * tileView.get(tileX - 1, tileY - 1);
				laplacianValue += 2 * tileView.get(tileX - 1, tileY + 1);
				laplacianValue += 2 * tileView.get(tileX + 1, tileY - 1);
				laplacianValue += 2 * tileView.get(tileX + 1, tileY + 1);
			}

			static_assert(MASK_DIM <= 3, "Laplacian algorithm for masks greater than 3x3 is not implemented yet!");
		}

		float* blockSumPtr = &blockSums[unifiedBlockIdx];
		blockReduceSum(laplacianValue * laplacianValue, unifiedThreadIdx, [blockSumPtr](int blockSum) {
			*blockSumPtr = blockSum;
		});
	}

	template<int MASK_DIM = 1>
		requires(MASK_DIM > 0 && MASK_DIM % 2 == 1)
	__global__ void laplacianSquareSumAlt(DataAccessor<const uchar> input, float* totalSum, Size size) {
		constexpr int HALO_WIDTH = (MASK_DIM > 1) ? (MASK_DIM - 1) / 2 : 1;
		constexpr int TILE_DIM = SQUARE_BLOCK_DIM + 2 * HALO_WIDTH;

		__shared__ int tile[TILE_DIM][TILE_DIM];
		TileView tileView{tile};

		const int unifiedThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;

		// Copy block and halo to shared memory
		constexpr int PIXELS_TO_COPY_BY_THREAD = (TILE_DIM * TILE_DIM + BLOCK_TOTAL_SIZE - 1) / (BLOCK_TOTAL_SIZE);

		for(int i = 0; i < PIXELS_TO_COPY_BY_THREAD; ++i) {
			const int tilePixelIdx = unifiedThreadIdx + i * BLOCK_TOTAL_SIZE;

			if(tilePixelIdx < TILE_DIM * TILE_DIM) {
				const int tilePixelX = tilePixelIdx % TILE_DIM;
				const int tilePixelY = tilePixelIdx / TILE_DIM;

				const int inputX = tilePixelX + blockIdx.x * blockDim.x - HALO_WIDTH;
				const int inputY = tilePixelY + blockIdx.y * blockDim.y - HALO_WIDTH;

				const bool inputInRange = isInRange(inputX, 0, size.width - 1) && isInRange(inputY, 0, size.height - 1);
				tileView.at(tilePixelX, tilePixelX) = inputInRange ? input.get(inputX, inputY) : 0;
			}
		}
		__syncthreads();

		// Compute Laplacian
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		const int tileX = threadIdx.x + HALO_WIDTH;
		const int tileY = threadIdx.y + HALO_WIDTH;

		int laplacianValue = 0;

		if(isInRange(x, HALO_WIDTH, size.width - 1 - HALO_WIDTH) && isInRange(y, HALO_WIDTH, size.height - 1 - HALO_WIDTH)) {
			if constexpr(MASK_DIM == 1) {
				laplacianValue = -4 * tileView.get(tileX, tileY);
				laplacianValue += tileView.get(tileX, tileY - 1);
				laplacianValue += tileView.get(tileX, tileY + 1);
				laplacianValue += tileView.get(tileX - 1, tileY);
				laplacianValue += tileView.get(tileX + 1, tileY);
			} else if constexpr(MASK_DIM == 3) {
				laplacianValue = -8 * tileView.get(tileX, tileY);
				laplacianValue += 2 * tileView.get(tileX - 1, tileY - 1);
				laplacianValue += 2 * tileView.get(tileX - 1, tileY + 1);
				laplacianValue += 2 * tileView.get(tileX + 1, tileY - 1);
				laplacianValue += 2 * tileView.get(tileX + 1, tileY + 1);
			}

			static_assert(MASK_DIM <= 3, "Laplacian algorithm for masks greater than 3x3 is not implemented yet!");
		}

		blockReduceSum(laplacianValue * laplacianValue, unifiedThreadIdx, [totalSum](int blockSum) {
			atomicAdd(totalSum, blockSum);
		});
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
