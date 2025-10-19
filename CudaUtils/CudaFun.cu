#include "CudaFun.cuh"

#ifdef HAS_CUDA
#	include <opencv4/opencv2/core/cuda_stream_accessor.hpp>
#endif

// Example CUDA device function to image color inversion
__device__ __forceinline__ uchar3 invertColor(uchar3 color) {
	return make_uchar3(255U - color.x, 255U - color.y, 255U - color.z);
}

// Example CUDA kernel to image color inversion
__global__ void processKernel(const uchar3* input, uchar3* output, int width, int height, int pitch) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < width && y < height) {
		const uchar3& inputPixel = ((const uchar3*)((const uchar*)input + y * pitch))[x];
		uchar3& outputPixel = ((uchar3*)((uchar*)output + y * pitch))[x];
		// USER CODE BEGIN
		outputPixel = invertColor(inputPixel);
		// USER CODE END
	}
}

// Example wrapper function to image color inversion
void processImageWithCuda(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
	const int width = src.cols;
	const int height = src.rows;
	const int pitch = src.step;

	const dim3 blockSize{16, 16};
	const dim3 gridSize{
		(width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y,
	};

	// Change allocation size if not match
	dst.create(src.size(), src.type());

	processKernel<<<gridSize, blockSize>>>(
		src.ptr<uchar3>(), dst.ptr<uchar3>(), width, height, pitch);
	cudaDeviceSynchronize();
}

// Example wrapper function to asynchronous image color inversion
void processImageWithCudaAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream) {
	const int width = src.cols;
	const int height = src.rows;
	const int pitch = src.step;

	const dim3 blockSize{16, 16};
	const dim3 gridSize{
		(width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y,
	};

	// Change allocation size if not match
	dst.create(src.size(), src.type());

	auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);
	processKernel<<<gridSize, blockSize, 0, cudaStream>>>(
		src.ptr<uchar3>(), dst.ptr<uchar3>(), width, height, pitch);
}
