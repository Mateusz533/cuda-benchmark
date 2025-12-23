#include "CudaUtils.cuh"
//
#ifdef HAS_CUDA
#	include <opencv2/core/cuda_stream_accessor.hpp>
#endif
//
#include "Kernels.cuh"

namespace CudaUtils
{
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

		invertColorKernel<<<gridSize, blockSize>>>(
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
		invertColorKernel<<<gridSize, blockSize, 0, cudaStream>>>(
			src.ptr<uchar3>(), dst.ptr<uchar3>(), width, height, pitch);
	}

	/* ==================================================================================================== */
}
