#pragma once
//
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
//
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/version.hpp>

namespace CudaUtils
{
	// Example wrapper function to image color inversion
	void processImageWithCuda(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

	// Example wrapper function to asynchronous image color inversion
	void processImageWithCudaAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream);

	/* ==================================================================================================== */

	void invertColor(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
	void invertColorAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream);
	void warpAffine(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Matx23f& transform);
	void warpAffineAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const cv::Matx23f& transform, cv::cuda::Stream& stream);

	/* ==================================================================================================== */

	class LaplaceRmsCalculator
	{
	public:
		LaplaceRmsCalculator();
		LaplaceRmsCalculator(const LaplaceRmsCalculator&) = delete;
		LaplaceRmsCalculator(LaplaceRmsCalculator&&) = delete;
		LaplaceRmsCalculator& operator=(const LaplaceRmsCalculator&) = delete;
		LaplaceRmsCalculator& operator=(LaplaceRmsCalculator&&) = delete;

		~LaplaceRmsCalculator();

		double Calculate(const cv::cuda::GpuMat& src, int kernelSize = 1, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

	private:
		float* dTotalSum;
		cv::cuda::GpuMat blockSums;
	};
}
