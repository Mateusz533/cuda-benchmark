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
	std::cout << "Test of calling `" << name << "`:" << std::endl;
	{
		using ResultType = std::invoke_result_t<Fun, Args&...>;
		auto callable = std::forward<Fun>(fun);

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

	constexpr auto copyTo = (void(cv::cuda::GpuMat::*)(cv::cuda::GpuMat&) const)(&cv::cuda::GpuMat::copyTo);
	runPerformanceTest<N>("copyTo (GRAY image)", copyTo, grayImg, tempDest);
	runPerformanceTest<N>("copyTo (BGR image)", copyTo, bgrImg, tempDest);
	runPerformanceTest<N>("copyTo (BGRA image)", copyTo, bgraImg, tempDest);
}
