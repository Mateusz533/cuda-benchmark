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
		// Assert no aliasing
		const cv::cuda::GpuMat& srcView = (&src == &dst) ? cv::cuda::GpuMat(src) : src;

		// Change allocation size if not match
		dst.create(srcView.size(), srcView.type());

		const int width = srcView.cols;
		const int height = srcView.rows;
		const int inPitch = srcView.step;
		const int outPitch = dst.step;

		const dim3 blockSize{16, 16};
		const dim3 gridSize{
			(width + blockSize.x - 1) / blockSize.x,
			(height + blockSize.y - 1) / blockSize.y,
		};

		invertColorKernel<<<gridSize, blockSize>>>(
			srcView.ptr<uchar3>(), dst.ptr<uchar3>(), width, height, inPitch, outPitch);
		cudaStreamSynchronize(0);
	}

	// Example wrapper function to asynchronous image color inversion
	void processImageWithCudaAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream) {
		// Assert no aliasing
		const cv::cuda::GpuMat& srcView = (&src == &dst) ? cv::cuda::GpuMat(src) : src;

		// Change allocation size if not match
		dst.create(srcView.size(), srcView.type());

		const int width = srcView.cols;
		const int height = srcView.rows;
		const int inPitch = srcView.step;
		const int outPitch = dst.step;

		const dim3 blockSize{16, 16};
		const dim3 gridSize{
			(width + blockSize.x - 1) / blockSize.x,
			(height + blockSize.y - 1) / blockSize.y,
		};

		const auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);
		invertColorKernel<<<gridSize, blockSize, 0, cudaStream>>>(
			srcView.ptr<uchar3>(), dst.ptr<uchar3>(), width, height, inPitch, outPitch);
	}

	/* ==================================================================================================== */
}
