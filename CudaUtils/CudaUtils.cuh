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
}
