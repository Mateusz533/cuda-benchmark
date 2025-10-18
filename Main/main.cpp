#include <chrono>
#include <iostream>
//
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
//
#include "CudaFun.cuh"

int main() {
	constexpr size_t N = 1'000;

	const cv::Mat demoImage = cv::imread("../Images/SrcImg.png");

	cv::cuda::GpuMat grayImg{};
	cv::cuda::GpuMat bgrImg{demoImage};
	cv::cuda::GpuMat bgraImg{};
	cv::cuda::cvtColor(bgrImg, grayImg, cv::COLOR_BGR2GRAY);
	cv::cuda::cvtColor(bgrImg, bgraImg, cv::COLOR_BGR2BGRA);

	cv::cuda::GpuMat tempDest;

	{
		std::cout << "Test of calling `processImageWithCuda`:" << std::endl;

		const auto startTime = std::chrono::high_resolution_clock::now();

		for(int i = 0; i < N; ++i)
			processImageWithCuda(bgrImg, tempDest);

		const auto endTime = std::chrono::high_resolution_clock::now();

		const auto totalTimeUs = std::chrono::duration<double, std::micro>(endTime - startTime).count();
		printf("- One call period: %8.3f us\n", totalTimeUs / N);
	}
}
