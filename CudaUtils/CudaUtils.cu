#include <cstddef>

#include "CudaUtils.cuh"
//
#ifdef HAS_CUDA
#	include <opencv2/core/cuda_stream_accessor.hpp>
#endif
//
#include "Kernels.cuh"

namespace CudaUtils
{
	void processImageWithCuda(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
		// Assert no aliasing
		const cv::cuda::GpuMat& srcView = (&src == &dst) ? cv::cuda::GpuMat(src) : src;

		// Change allocation size if not match
		dst.create(srcView.size(), srcView.type());

		const int width = srcView.cols;
		const int height = srcView.rows;
		const std::size_t inPitch = srcView.step;
		const std::size_t outPitch = dst.step;

		const dim3 blockSize{16, 16};
		const dim3 gridSize{
			(width + blockSize.x - 1) / blockSize.x,
			(height + blockSize.y - 1) / blockSize.y,
		};

		processImageKernel<<<gridSize, blockSize>>>(
			srcView.ptr<uchar3>(), dst.ptr<uchar3>(), width, height, inPitch, outPitch);
		cudaStreamSynchronize(0);
	}

	void processImageWithCudaAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream) {
		// Assert no aliasing
		const cv::cuda::GpuMat& srcView = (&src == &dst) ? cv::cuda::GpuMat(src) : src;

		// Change allocation size if not match
		dst.create(srcView.size(), srcView.type());

		const int width = srcView.cols;
		const int height = srcView.rows;
		const std::size_t inPitch = srcView.step;
		const std::size_t outPitch = dst.step;

		const dim3 blockSize{16, 16};
		const dim3 gridSize{
			(width + blockSize.x - 1) / blockSize.x,
			(height + blockSize.y - 1) / blockSize.y,
		};

		const auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);
		processImageKernel<<<gridSize, blockSize, 0, cudaStream>>>(
			srcView.ptr<uchar3>(), dst.ptr<uchar3>(), width, height, inPitch, outPitch);
	}

	/* ==================================================================================================== */

	template<typename PixelType, typename Operation>
	void unaryOperation(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
		// Assert no aliasing
		const cv::cuda::GpuMat& srcView = (&src == &dst) ? cv::cuda::GpuMat(src) : src;

		// Change allocation size if not match
		dst.create(srcView.size(), srcView.type());

		const Size size{srcView.cols, srcView.rows};
		const DataAccessor input{srcView.ptr<PixelType>(), srcView.step};
		const DataAccessor output{dst.ptr<PixelType>(), dst.step};

		const dim3 blockSize{16, 16};
		const dim3 gridSize{
			(size.width + blockSize.x - 1) / blockSize.x,
			(size.height + blockSize.y - 1) / blockSize.y,
		};

		unaryOperationKernel<PixelType, Operation><<<gridSize, blockSize>>>(input, output, size);
		cudaStreamSynchronize(0);
	}

	template<typename PixelType, typename Operation>
	void unaryOperationAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream) {
		// Assert no aliasing
		const cv::cuda::GpuMat& srcView = (&src == &dst) ? cv::cuda::GpuMat(src) : src;

		// Change allocation size if not match
		dst.create(srcView.size(), srcView.type());

		const Size size{srcView.cols, srcView.rows};
		const DataAccessor input{srcView.ptr<PixelType>(), srcView.step};
		const DataAccessor output{dst.ptr<PixelType>(), dst.step};

		const dim3 blockSize{16, 16};
		const dim3 gridSize{
			(size.width + blockSize.x - 1) / blockSize.x,
			(size.height + blockSize.y - 1) / blockSize.y,
		};

		const auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);
		unaryOperationKernel<PixelType, Operation><<<gridSize, blockSize, 0, cudaStream>>>(input, output, size);
	}

	void invertColor(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
		if(src.channels() == 1) {
			unaryOperation<uchar1, InvertColor<uchar1>>(src, dst);
		} else if(src.channels() == 3) {
			unaryOperation<uchar3, InvertColor<uchar3>>(src, dst);
		} else if(src.channels() == 4) {
			unaryOperation<uchar4, InvertColor<uchar4>>(src, dst);
		} else {
			// Error !
		}
	}

	void invertColorAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream) {
		if(src.channels() == 1) {
			unaryOperationAsync<uchar1, InvertColor<uchar1>>(src, dst, stream);
		} else if(src.channels() == 3) {
			unaryOperationAsync<uchar3, InvertColor<uchar3>>(src, dst, stream);
		} else if(src.channels() == 4) {
			unaryOperationAsync<uchar4, InvertColor<uchar4>>(src, dst, stream);
		} else {
			// Error !
		}
	}
}
