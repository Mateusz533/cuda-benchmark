#include <chrono>
#include <iostream>
//
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
//
#include "CudaFun.cuh"

template<size_t N, typename Fun, typename... Args>
inline void runPerformanceTest(const std::string& name, Fun&& fun, Args&... args) {
	std::cout << "Test of calling synchronous function \033[34m`" << name << "`\033[0m:" << std::endl;
	{
		using ResultType = std::invoke_result_t<Fun, Args&...>;
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
inline void runPerformanceTestWithStream(const std::string& name, Fun&& fun, Args&... args) {
	std::cout << "Test of calling asynchronous function \033[32m`" << name << "`\033[0m:" << std::endl;
	{
		using ResultType = std::invoke_result_t<Fun, Args&..., cv::cuda::Stream&>;
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

	cv::cuda::GpuMat tempDest;

	/* ======================================================================================================================== */

	runPerformanceTest<N>("processImageWithCuda (BGR image)", processImageWithCuda, bgrImg, tempDest);
	runPerformanceTestWithStream<N>("processImageWithCudaAsync (BGR image)", processImageWithCudaAsync, bgrImg, tempDest);

	constexpr auto syncCopyTo = (void(cv::cuda::GpuMat::*)(cv::cuda::GpuMat&) const)(&cv::cuda::GpuMat::copyTo);
	runPerformanceTest<N>("copyTo (GRAY image)", syncCopyTo, grayImg, tempDest);
	runPerformanceTest<N>("copyTo (BGR  image)", syncCopyTo, bgrImg, tempDest);
	runPerformanceTest<N>("copyTo (BGRA image)", syncCopyTo, bgraImg, tempDest);

	constexpr auto asyncCopyTo = (void(cv::cuda::GpuMat::*)(cv::cuda::GpuMat&, cv::cuda::Stream&) const)(&cv::cuda::GpuMat::copyTo);
	runPerformanceTestWithStream<N>("copyTo (GRAY image)", asyncCopyTo, grayImg, tempDest);
	runPerformanceTestWithStream<N>("copyTo (BGR  image)", asyncCopyTo, bgrImg, tempDest);
	runPerformanceTestWithStream<N>("copyTo (BGRA image)", asyncCopyTo, bgraImg, tempDest);
}
