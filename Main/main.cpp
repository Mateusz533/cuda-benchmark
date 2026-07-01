#include <chrono>
#include <iostream>
//
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
//
#include "CudaUtils.cuh"

template<size_t N, typename Fun, typename... Args>
inline void runPerformanceTest(const std::string& name, Fun&& fun, Args&&... args) {
	std::cout << "Test of calling synchronous function \033[34m`" << name << "`\033[0m:" << std::endl;
	{
		using ResultType = std::invoke_result_t<Fun, Args&&...>;
		auto callable = std::forward<Fun>(fun);

		// Do not measure initialization time
		std::invoke(callable, args...);

		const auto startTime = std::chrono::high_resolution_clock::now();

		for(int i = 0; i < N; ++i) {
			std::invoke(callable, args...);
		}

		const auto endTime = std::chrono::high_resolution_clock::now();

		const auto totalTimeUs = std::chrono::duration<double, std::micro>(endTime - startTime).count();
		printf("- One call period: %8.3f us\n", totalTimeUs / N);

		if constexpr(!std::is_void_v<ResultType>) {
			const auto lastResult = std::invoke(callable, args...);

			std::cout << "- Last result: ";
			if constexpr(std::is_same_v<ResultType, bool>)
				std::cout << std::boolalpha;
			std::cout << lastResult << std::endl;
		}
	}
	std::cout << std::endl;
}

template<size_t N, typename Fun, typename... Args>
inline void runPerformanceTestWithStream(const std::string& name, Fun&& fun, Args&&... args) {
	std::cout << "Test of calling asynchronous function \033[32m`" << name << "`\033[0m:" << std::endl;
	{
		using ResultType = std::invoke_result_t<Fun, Args&&..., cv::cuda::Stream&>;
		auto callable = std::forward<Fun>(fun);

		cv::cuda::Stream stream;
		cv::cuda::Event startEvent{cv::cuda::Event::CreateFlags::BLOCKING_SYNC};
		cv::cuda::Event endEvent{cv::cuda::Event::CreateFlags::BLOCKING_SYNC};

		// Do not measure initialization time
		std::invoke(callable, args..., stream);
		stream.waitForCompletion();

		const auto startTime = std::chrono::high_resolution_clock::now();
		startEvent.record(stream);

		for(int i = 0; i < N; ++i) {
			std::invoke(callable, args..., stream);
		}

		const auto asyncEndTime = std::chrono::high_resolution_clock::now();
		endEvent.record(stream);

		stream.waitForCompletion();
		const auto syncEndTime = std::chrono::high_resolution_clock::now();

		const auto totalAsyncTimeUs = std::chrono::duration<double, std::micro>(asyncEndTime - startTime).count();
		const auto totalChronoTimeUs = std::chrono::duration<double, std::micro>(syncEndTime - startTime).count();
		const auto totalEventsTimeUs = cv::cuda::Event::elapsedTime(startEvent, endEvent) * 1000.0;

		printf("- One asynchronous call period : %8.3f us\n", totalAsyncTimeUs / N);
		printf("- One operation period (chrono): %8.3f us\n", totalChronoTimeUs / N);
		printf("- One operation period (events): %8.3f us\n", totalEventsTimeUs / N);

		if constexpr(!std::is_void_v<ResultType>) {
			const auto lastResult = std::invoke(callable, args..., stream);

			std::cout << "- Last result: ";
			if constexpr(std::is_same_v<ResultType, bool>)
				std::cout << std::boolalpha;
			std::cout << lastResult << std::endl;
		}
	}
	std::cout << std::endl;
}

int main() {
	constexpr size_t N = 1'000;

	const cv::Mat demoImage = cv::imread("../Images/SrcImg.png");

	cv::cuda::GpuMat grayImg{};
	cv::cuda::GpuMat bgrImg{demoImage};
	cv::cuda::GpuMat bgraImg{};
	cv::cuda::cvtColor(bgrImg, grayImg, cv::COLOR_BGR2GRAY);
	cv::cuda::cvtColor(bgrImg, bgraImg, cv::COLOR_BGR2BGRA);

	cv::Matx23d affineTransform = {
		+0.8, -0.6, +0.1 * grayImg.cols + 0.3 * grayImg.rows,
		+0.6, +0.8, -0.3 * grayImg.cols + 0.1 * grayImg.rows};

	cv::Mat maxLaplacianImageHost(demoImage.rows, demoImage.cols, CV_8UC1);
	for(int x = 0; x < demoImage.cols; ++x) {
		for(int y = 0; y < demoImage.rows; ++y) {
			maxLaplacianImageHost.at<uchar>(y, x) = 255 * ((x + y) % 2);
		}
	}

	cv::cuda::GpuMat maxLaplacianImage{maxLaplacianImageHost};
	const auto laplaceFilterGray = cv::cuda::createLaplacianFilter(CV_32FC1, CV_32FC1, 1);

	const auto syncLaplacianMeanSqrFunc = [&laplaceFilterGray](const cv::cuda::GpuMat& src) {
		constexpr int MAX_PIXEL_VALUE = (4 * 255) * (4 * 255);
		cv::cuda::GpuMat srcFloat, tempDst;
		src.convertTo(srcFloat, CV_32F);
		laplaceFilterGray.get()->apply(srcFloat, tempDst);
		return cv::cuda::sqrSum(tempDst) / tempDst.size().area() / MAX_PIXEL_VALUE;
	};

	auto laplaceRmsCalculator = CudaUtils::LaplaceRmsCalculator();

	cv::cuda::GpuMat tempDest;

	/* ======================================================================================================================== */

	runPerformanceTest<N>("processImageWithCuda (BGR image)", CudaUtils::processImageWithCuda, bgrImg, tempDest);
	runPerformanceTestWithStream<N>("processImageWithCudaAsync (BGR image)", CudaUtils::processImageWithCudaAsync, bgrImg, tempDest);

	/* ======================================================================================================================== */

	runPerformanceTest<N>("invertColor (GRAY image)", CudaUtils::invertColor, grayImg, tempDest);
	runPerformanceTest<N>("invertColor (BGR  image)", CudaUtils::invertColor, bgrImg, tempDest);
	runPerformanceTest<N>("invertColor (BGRA image)", CudaUtils::invertColor, bgraImg, tempDest);

	runPerformanceTestWithStream<N>("invertColorAsync (GRAY image)", CudaUtils::invertColorAsync, grayImg, tempDest);
	runPerformanceTestWithStream<N>("invertColorAsync (BGR  image)", CudaUtils::invertColorAsync, bgrImg, tempDest);
	runPerformanceTestWithStream<N>("invertColorAsync (BGRA image)", CudaUtils::invertColorAsync, bgraImg, tempDest);

	runPerformanceTest<N>("warpAffine (GRAY image)", CudaUtils::warpAffine, grayImg, tempDest, affineTransform);
	runPerformanceTest<N>("warpAffine (BGR  image)", CudaUtils::warpAffine, bgrImg, tempDest, affineTransform);
	runPerformanceTest<N>("warpAffine (BGRA image)", CudaUtils::warpAffine, bgraImg, tempDest, affineTransform);

	runPerformanceTestWithStream<N>("warpAffineAsync (GRAY image)", CudaUtils::warpAffineAsync, grayImg, tempDest, affineTransform);
	runPerformanceTestWithStream<N>("warpAffineAsync (BGR  image)", CudaUtils::warpAffineAsync, bgrImg, tempDest, affineTransform);
	runPerformanceTestWithStream<N>("warpAffineAsync (BGRA image)", CudaUtils::warpAffineAsync, bgraImg, tempDest, affineTransform);

	constexpr auto syncCopyTo = static_cast<void (cv::cuda::GpuMat::*)(cv::cuda::GpuMat&) const>(&cv::cuda::GpuMat::copyTo);
	runPerformanceTest<N>("copyTo (GRAY image)", syncCopyTo, grayImg, tempDest);
	runPerformanceTest<N>("copyTo (BGR  image)", syncCopyTo, bgrImg, tempDest);
	runPerformanceTest<N>("copyTo (BGRA image)", syncCopyTo, bgraImg, tempDest);

	constexpr auto asyncCopyTo = static_cast<void (cv::cuda::GpuMat::*)(cv::cuda::GpuMat&, cv::cuda::Stream&) const>(&cv::cuda::GpuMat::copyTo);
	runPerformanceTestWithStream<N>("copyTo (GRAY image)", asyncCopyTo, grayImg, tempDest);
	runPerformanceTestWithStream<N>("copyTo (BGR  image)", asyncCopyTo, bgrImg, tempDest);
	runPerformanceTestWithStream<N>("copyTo (BGRA image)", asyncCopyTo, bgraImg, tempDest);

	constexpr auto asyncWarpAffine = static_cast<void (*)(cv::InputArray, cv::OutputArray, cv::InputArray, cv::Size, int, int, cv::Scalar, cv::cuda::Stream&)>(&cv::cuda::warpAffine);
	runPerformanceTestWithStream<N>("warpAffine (GRAY image)", asyncWarpAffine, grayImg, tempDest, affineTransform, grayImg.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	runPerformanceTestWithStream<N>("warpAffine (BGR  image)", asyncWarpAffine, bgrImg, tempDest, affineTransform, bgrImg.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	runPerformanceTestWithStream<N>("warpAffine (BGRA image)", asyncWarpAffine, bgraImg, tempDest, affineTransform, bgraImg.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	runPerformanceTest<N>("laplacianMeanSqr (Real  image)", syncLaplacianMeanSqrFunc, grayImg);
	runPerformanceTest<N>("laplacianMeanSqr (Ideal image)", syncLaplacianMeanSqrFunc, maxLaplacianImage);

	runPerformanceTestWithStream<N>("laplacianRms (Real  image)", &CudaUtils::LaplaceRmsCalculator::Calculate, laplaceRmsCalculator, grayImg, 1);
	runPerformanceTestWithStream<N>("laplacianRms (Ideal image)", &CudaUtils::LaplaceRmsCalculator::Calculate, laplaceRmsCalculator, maxLaplacianImage, 1);
}
