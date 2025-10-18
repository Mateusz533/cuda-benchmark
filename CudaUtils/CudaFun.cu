#include "CudaFun.cuh"

// Example CUDA device function to image color inversion
__device__ __forceinline__ uchar3 invertColor(uchar3 color) {
	return make_uchar3(255U - color.x, 255U - color.y, 255U - color.z);
}

// Example CUDA kernel to image color inversion
__global__ void processKernel(const uchar3* input, uchar3* output, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < width && y < height) {
		const int idx = y * width + x;
		// USER CODE BEGIN
		output[idx] = invertColor(input[idx]);
		// USER CODE END
	}
}

// Example wrapper function to image color inversion
void processImageWithCuda(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
	const int width = src.cols;
	const int height = src.rows;

	const dim3 blockSize(16, 16);
	const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
						(height + blockSize.y - 1) / blockSize.y);

	if(&src == &dst) {
		cv::cuda::GpuMat result = cv::cuda::createContinuous(src.size(), src.type());
		processKernel<<<gridSize, blockSize>>>(src.ptr<uchar3>(),
											   result.ptr<uchar3>(), width, height);
		cudaDeviceSynchronize();
		dst = std::move(result);
	} else {
		processKernel<<<gridSize, blockSize>>>(src.ptr<uchar3>(), dst.ptr<uchar3>(),
											   width, height);
		cudaDeviceSynchronize();
	}
}
