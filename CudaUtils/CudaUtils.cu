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

		const dim3 blockSize{SQUARE_BLOCK_DIM, SQUARE_BLOCK_DIM};
		const dim3 gridSize{
			(width + blockSize.x - 1) / blockSize.x,
			(height + blockSize.y - 1) / blockSize.y,
		};

		Kernels::processImage<<<gridSize, blockSize>>>(
			srcView.ptr<uchar3>(), dst.ptr<uchar3>(), width, height, inPitch, outPitch);
		cudaStreamSynchronize(nullptr);
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

		const dim3 blockSize{SQUARE_BLOCK_DIM, SQUARE_BLOCK_DIM};
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

		const dim3 blockSize{SQUARE_BLOCK_DIM, SQUARE_BLOCK_DIM};
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

		const dim3 blockSize{SQUARE_BLOCK_DIM, SQUARE_BLOCK_DIM};
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

	/* ==================================================================================================== */

	LaplaceRmsCalculator::LaplaceRmsCalculator() {
		cudaMalloc(&dTotalSum, sizeof(float));
	}

	LaplaceRmsCalculator::~LaplaceRmsCalculator() {
		cudaFree(dTotalSum);
	}

	double LaplaceRmsCalculator::Calculate(const cv::cuda::GpuMat& src, int kernelSize, cv::cuda::Stream& stream) {
		if(kernelSize != 1 && kernelSize != 3) {
			return -1.0;
		}

		if(src.type() != CV_8U || src.cols < kernelSize || src.rows < kernelSize) {
			return -1.0;
		}

		const Size size{src.cols, src.rows};
		const DataAccessor input{src.ptr<uchar>(), src.step};

		const dim3 blockSize{SQUARE_BLOCK_DIM, SQUARE_BLOCK_DIM};
		const dim3 gridSize{
			(size.width + blockSize.x - 1) / blockSize.x,
			(size.height + blockSize.y - 1) / blockSize.y,
		};

		// Change allocation size if not match
		const int blockCount = gridSize.x * gridSize.y;
		blockSums.create(1, blockCount, CV_32F);

		const auto cudaStream = cv::cuda::StreamAccessor::getStream(stream);

		cudaMemsetAsync(dTotalSum, 0, sizeof(float), cudaStream);
		auto* blockSumsPtr = blockSums.ptr<float>();

		if(kernelSize == 1) {
			Kernels::laplacianSquareSum<1><<<gridSize, blockSize, 0, cudaStream>>>(input, blockSumsPtr, size);
		} else if(kernelSize == 3) {
			Kernels::laplacianSquareSum<3><<<gridSize, blockSize, 0, cudaStream>>>(input, blockSumsPtr, size);
		} else {
			// Error handling
		}

		{
			const dim3 blockSize{BLOCK_TOTAL_SIZE};
			const dim3 gridSize{(blockCount + blockSize.x - 1) / blockSize.x};
			Kernels::reduceSum<<<gridSize, blockSize, 0, cudaStream>>>(blockSumsPtr, dTotalSum, blockCount);
		}

		float hTotalSum = 1.0;
		cudaMemcpyAsync(&hTotalSum, dTotalSum, sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
		stream.waitForCompletion();

		const long maskDim = (kernelSize == 1 ? 3 : 3);	 // More option after more variants implementation
		const long kernelMultiplier = (kernelSize == 1 ? 4 : 8);
		const long maxPixelValue = std::pow(kernelMultiplier * 255L, 2L);

		const long effectiveArea = (size.width - maskDim + 1) * (size.height - maskDim + 1);
		const double variance = double(hTotalSum) / (maxPixelValue * effectiveArea);

		return std::sqrt(variance);
	}
}
