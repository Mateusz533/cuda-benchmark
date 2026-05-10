#pragma once
//
#include <type_traits>
//
#ifdef HAS_CUDA
#	include <cuda_runtime.h>
#endif
//
#include <opencv2/core/hal/interface.h>

namespace CudaUtils
{
	// Example CUDA device function to image color inversion
	__device__ __forceinline__ uchar3 processImage(uchar3 color) {
		return make_uchar3(255U - color.x, 255U - color.y, 255U - color.z);
	}

	// Example CUDA device functor to image color inversion
	struct ProcessImage {
		__device__ __forceinline__ uchar3 operator()(uchar3 color) { return processImage(color); }

	private:
		[[no_unique_address]] const char nothing[0]{};
	};

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

	/* ==================================================================================================== */

	template<typename PixelType>
		requires(std::is_same_v<PixelType, uchar1> || std::is_same_v<PixelType, uchar3> || std::is_same_v<PixelType, uchar4>)
	__device__ __forceinline__ PixelType invertColor(PixelType color) {
		if constexpr(std::is_same_v<PixelType, uchar1>) {
			return make_uchar1(255U - color.x);
		} else if constexpr(std::is_same_v<PixelType, uchar3>) {
			return make_uchar3(255U - color.x, 255U - color.y, 255U - color.z);
		} else if constexpr(std::is_same_v<PixelType, uchar4>) {
			return make_uchar4(255U - color.x, 255U - color.y, 255U - color.z, 255U - color.w);
		}
	}

	template<typename PixelType>
	struct InvertColor {
		__device__ __forceinline__ PixelType operator()(PixelType color) { return invertColor(color); }

	private:
		[[no_unique_address]] const char nothing[0]{};
	};
}
