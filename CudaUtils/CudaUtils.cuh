#pragma once
//
#include <opencv4/opencv2/core/cvdef.h>
#include <opencv4/opencv2/core/hal/interface.h>
//
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/core/version.hpp>

namespace CudaUtils
{
	// Declaration of functions from file .cu
	void processImageWithCuda(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
	void processImageWithCudaAsync(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream);
}
