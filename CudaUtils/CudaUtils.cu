#include "CudaUtils.cuh"
//
#ifdef HAS_CUDA
#	include <opencv2/core/cuda_stream_accessor.hpp>
#endif
//
#include <opencv2/imgproc.hpp>
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

		const dim3 blockSize{BLOCK_DIM, BLOCK_DIM};
		const dim3 gridSize{
			(width + blockSize.x - 1) / blockSize.x,
			(height + blockSize.y - 1) / blockSize.y,
		};

		Kernels::processImage<<<gridSize, blockSize>>>(
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

		const dim3 blockSize{BLOCK_DIM, BLOCK_DIM};
		const dim3 gridSize{
			(width + blockSize.x - 1) / blockSize.x,
			(height + blockSize.y - 1) / blockSize.y,
		};

		const auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);
		Kernels::processImage<<<gridSize, blockSize, 0, cudaStream>>>(
			srcView.ptr<uchar3>(), dst.ptr<uchar3>(), width, height, inPitch, outPitch);
	}

	/* ==================================================================================================== */

	template<typename PixelType, typename Operation>
	void unaryOperationAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream) {
		// Assert no aliasing
		const cv::cuda::GpuMat& srcView = (&src == &dst) ? cv::cuda::GpuMat(src) : src;

		// Change allocation size if not match
		dst.create(srcView.size(), srcView.type());

		const Size size{srcView.cols, srcView.rows};
		const DataAccessor input{srcView.ptr<PixelType>(), srcView.step};
		const DataAccessor output{dst.ptr<PixelType>(), dst.step};

		const dim3 blockSize{BLOCK_DIM, BLOCK_DIM};
		const dim3 gridSize{
			(size.width + blockSize.x - 1) / blockSize.x,
			(size.height + blockSize.y - 1) / blockSize.y,
		};

		const auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);
		Kernels::unaryOperation<PixelType, Operation><<<gridSize, blockSize, 0, cudaStream>>>(input, output, size);
	}

	template<typename PixelType, typename Operation, typename... Args>
	void nonlinearInputUnaryOperationAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream, Args... args) {
		// Assert no aliasing
		const cv::cuda::GpuMat& srcView = (&src == &dst) ? cv::cuda::GpuMat(src) : src;

		// Change allocation size if not match
		dst.create(srcView.size(), srcView.type());

		const Size size{srcView.cols, srcView.rows};
		const DataAccessor input{srcView.ptr<PixelType>(), srcView.step};
		const DataAccessor output{dst.ptr<PixelType>(), dst.step};

		const dim3 blockSize{BLOCK_DIM, BLOCK_DIM};
		const dim3 gridSize{
			(size.width + blockSize.x - 1) / blockSize.x,
			(size.height + blockSize.y - 1) / blockSize.y,
		};

		const auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);
		Kernels::nonlinearInputUnaryOperation<PixelType, Operation><<<gridSize, blockSize, 0, cudaStream>>>(input, output, size, args...);
	}

	void invertColorAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream) {
		if(src.channels() == 1) {
			unaryOperationAsync<uchar1, InvertColor<uchar1>>(src, dst, stream);
		} else if(src.channels() == 3) {
			unaryOperationAsync<uchar3, InvertColor<uchar3>>(src, dst, stream);
		} else if(src.channels() == 4) {
			unaryOperationAsync<uchar4, InvertColor<uchar4>>(src, dst, stream);
		} else {
			// Error handling
		}
	}

	void invertColor(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
		auto defaultStream = cv::cuda::Stream::Null();
		invertColorAsync(src, dst, defaultStream);
		defaultStream.waitForCompletion();
	}

	/* ==================================================================================================== */

	void warpAffineAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Matx23f& transform, cv::cuda::Stream& stream) {
		cv::Mat invMatrix;
		cv::invertAffineTransform(transform, invMatrix);
		const Matrix<float, 2, 3> invTransform{
			static_cast<float>(invMatrix.at<double>(0, 0)),
			static_cast<float>(invMatrix.at<double>(0, 1)),
			static_cast<float>(invMatrix.at<double>(0, 2)),
			static_cast<float>(invMatrix.at<double>(1, 0)),
			static_cast<float>(invMatrix.at<double>(1, 1)),
			static_cast<float>(invMatrix.at<double>(1, 2)),
		};

		if(src.channels() == 1) {
			nonlinearInputUnaryOperationAsync<uchar1, WarpAffine<uchar1>>(src, dst, stream, invTransform);
		} else if(src.channels() == 3) {
			nonlinearInputUnaryOperationAsync<uchar3, WarpAffine<uchar3>>(src, dst, stream, invTransform);
		} else if(src.channels() == 4) {
			nonlinearInputUnaryOperationAsync<uchar4, WarpAffine<uchar4>>(src, dst, stream, invTransform);
		} else {
			// Error handling
		}
	}

	void warpAffine(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Matx23f& transform) {
		auto defaultStream = cv::cuda::Stream::Null();
		warpAffineAsync(src, dst, transform, defaultStream);
		defaultStream.waitForCompletion();
	}
}
