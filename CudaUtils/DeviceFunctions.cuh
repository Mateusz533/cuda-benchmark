#pragma once
//
#ifdef HAS_CUDA
#	include <cuda_runtime.h>
#endif
//
#include <opencv2/core/hal/interface.h>

namespace CudaUtils
{
	// Example CUDA device function to image color inversion
	__device__ __forceinline__ uchar3 invertColor(uchar3 color) {
		return make_uchar3(255U - color.x, 255U - color.y, 255U - color.z);
	}

	/* ==================================================================================================== */

	template<typename T>
	__device__ __forceinline__ T& at(T* array, int pitchBytes, int x, int y) {
		return ((T*)((uchar*)array + y * pitchBytes))[x];
	}

	template<typename T>
	__device__ __forceinline__ const T& get(const T* array, int pitchBytes, int x, int y) {
		return ((const T*)((const uchar*)array + y * pitchBytes))[x];
	}

	template<typename T>
	__device__ __forceinline__ T warpReduceSum(T val) {
		for(int offset = warpSize / 2; offset > 0; offset /= 2) {
			val += __shfl_down_sync(0xffffffff, val, offset);
		}
		return val;
	}

	template<typename T>
	__device__ __forceinline__ T isInRange(T v, T minVal, T maxVal) {
		return minVal <= v && v <= maxVal;
	}

	template<typename T>
	__device__ __forceinline__ T clamp(T v, T minVal, T maxVal) {
		return v < minVal ? minVal : (v > maxVal ? maxVal : v);
	}
}
