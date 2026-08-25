#pragma once
//
#include <cstddef>
#include <type_traits>
//
#ifdef HAS_CUDA
#	include <cuda_runtime.h>
#endif
//
#include <opencv2/core/hal/interface.h>

namespace CudaUtils
{
	constexpr int SQUARE_BLOCK_DIM = 16;
	constexpr int BLOCK_TOTAL_SIZE = SQUARE_BLOCK_DIM * SQUARE_BLOCK_DIM;
	constexpr int WARP_SIZE = 32;  // See `warpSize` in `<__clang_cuda_builtin_vars.h>`

	struct Empty {
	private:
		[[no_unique_address]] const char nothing[0]{};
	};

	/* ==================================================================================================== */

	// Example CUDA device function to image color inversion
	__device__ __forceinline__ uchar3 processImage(uchar3 color) {
		return make_uchar3(255U - color.x, 255U - color.y, 255U - color.z);
	}

	// Example CUDA device functor to image color inversion
	struct ProcessImage : private Empty {
		__device__ __forceinline__ uchar3 operator()(uchar3 color) { return processImage(color); }
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
	__device__ __forceinline__ T isInRange(T v, T minVal, T maxVal) {
		return minVal <= v && v <= maxVal;
	}

	template<typename T>
	__device__ __forceinline__ T clamp(T v, T minVal, T maxVal) {
		return v < minVal ? minVal : (v > maxVal ? maxVal : v);
	}

	template<typename T>
	__device__ __forceinline__ T warpReduceSum(T val) {
		for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
			val += __shfl_down_sync(0xffffffff, val, offset);
		}
		return val;
	}

	template<typename T, typename Func>
	__device__ __forceinline__ void blockReduceSum(T inputValue, int threadId, Func applySumFunc) {
		constexpr int WARPS_PER_BLOCK = BLOCK_TOTAL_SIZE / WARP_SIZE;
		__shared__ T sharedSum[WARPS_PER_BLOCK];

		const int warpId = threadId / WARP_SIZE;
		const T warpSum = warpReduceSum(inputValue);

		if(threadId % WARP_SIZE == 0) {
			sharedSum[warpId] = warpSum;
		}
		__syncthreads();

		if(warpId == 0) {
			const T warpSum = (threadId < WARPS_PER_BLOCK) ? sharedSum[threadId] : T(0);
			const T blockSum = warpReduceSum(warpSum);
			if(threadId == 0) {
				applySumFunc(blockSum);
			}
		}
	}

	/* ==================================================================================================== */

	template<typename T>
	concept PixelLike =
		std::is_same_v<T, char1> || std::is_same_v<T, char2> || std::is_same_v<T, char3> || std::is_same_v<T, char4> ||
		std::is_same_v<T, uchar1> || std::is_same_v<T, uchar2> || std::is_same_v<T, uchar3> || std::is_same_v<T, uchar4> ||
		std::is_same_v<T, ushort1> || std::is_same_v<T, ushort2> || std::is_same_v<T, ushort3> || std::is_same_v<T, ushort4> ||
		std::is_same_v<T, short1> || std::is_same_v<T, short2> || std::is_same_v<T, short3> || std::is_same_v<T, short4> ||
		std::is_same_v<T, int1> || std::is_same_v<T, int2> || std::is_same_v<T, int3> || std::is_same_v<T, int4> ||
		std::is_same_v<T, uint1> || std::is_same_v<T, uint2> || std::is_same_v<T, uint3> || std::is_same_v<T, uint4> ||
		std::is_same_v<T, long1> || std::is_same_v<T, long2> || std::is_same_v<T, long3> || std::is_same_v<T, long4> ||
		std::is_same_v<T, ulong1> || std::is_same_v<T, ulong2> || std::is_same_v<T, ulong3> || std::is_same_v<T, ulong4> ||
		std::is_same_v<T, float1> || std::is_same_v<T, float2> || std::is_same_v<T, float3> || std::is_same_v<T, float4> ||
		std::is_same_v<T, double1> || std::is_same_v<T, double2> || std::is_same_v<T, double3> || std::is_same_v<T, double4>;

	template<PixelLike T>
	constexpr std::size_t CHANNELS = sizeof(T) / sizeof(std::declval<T>().x);

	template<PixelLike T>
	using ChannelType = decltype(std::declval<T>().x);

	struct Size {
		int width{};
		int height{};
	};

	template<typename PixelType>
	class DataAccessor
	{
	public:
		__host__ __device__ __forceinline__ DataAccessor(PixelType* data, std::size_t pitchBytes)
			: data{data}, pitchBytes{pitchBytes} {}

		__device__ __forceinline__ PixelType& at(int x, int y) const
			requires(!std::is_const_v<PixelType>)
		{
			return reinterpret_cast<PixelType*>(reinterpret_cast<uchar*>(data) + y * pitchBytes)[x];
		}

		__device__ __forceinline__ const PixelType& get(int x, int y) const {
			return reinterpret_cast<const PixelType*>(reinterpret_cast<const uchar*>(data) + y * pitchBytes)[x];
		}

	private:
		PixelType* data{};
		std::size_t pitchBytes{};
	};

	template<typename T, std::size_t N, std::size_t M>
	struct Matrix {
		T data[N][M];
	};

	template<typename Pixel, std::size_t DIM>
	class TileView
	{
	public:
		__device__ __forceinline__ TileView(Pixel (&tile)[DIM][DIM]) : tile{tile} {}

		__device__ __forceinline__ Pixel& at(int x, int y) {
			return tile[y][x];
		}

		__device__ __forceinline__ const Pixel& get(int x, int y) const {
			return tile[y][x];
		}

		__device__ __forceinline__ static Size size() {
			return Size{DIM, DIM};
		}

		__device__ __forceinline__ static int area() {
			return DIM * DIM;
		}

	private:
		Pixel (&tile)[DIM][DIM];
	};

	/* ==================================================================================================== */

	template<PixelLike PixelType>
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

	template<PixelLike PixelType>
	struct InvertColor : private Empty {
		__device__ __forceinline__ PixelType operator()(PixelType color) { return invertColor(color); }
	};

	/* ==================================================================================================== */

	template<PixelLike PixelType>
	__device__ PixelType bilinearInterpolate(DataAccessor<const PixelType> input, Size size, float x, float y) {
		x = clamp(x, 0.0f, static_cast<float>(size.width - 1));
		y = clamp(y, 0.0f, static_cast<float>(size.height - 1));

		const int x0 = static_cast<int>(floorf(x));
		const int y0 = static_cast<int>(floorf(y));
		const int x1 = min(x0 + 1, size.width - 1);
		const int y1 = min(y0 + 1, size.height - 1);

		const float dx = x - static_cast<float>(x0);
		const float dy = y - static_cast<float>(y0);

		const PixelType p00 = input.get(x0, y0);
		const PixelType p10 = input.get(x1, y0);
		const PixelType p01 = input.get(x0, y1);
		const PixelType p11 = input.get(x1, y1);

		const float w00 = (1.0f - dx) * (1.0f - dy);
		const float w10 = dx * (1.0f - dy);
		const float w01 = (1.0f - dx) * dy;
		const float w11 = dx * dy;

		using Type = ChannelType<PixelType>;
		PixelType result;

		for(std::size_t i = 0; i < CHANNELS<PixelType>; ++i) {
			const float value = w00 * reinterpret_cast<const Type*>(&p00)[i] +
								w10 * reinterpret_cast<const Type*>(&p10)[i] +
								w01 * reinterpret_cast<const Type*>(&p01)[i] +
								w11 * reinterpret_cast<const Type*>(&p11)[i];

			reinterpret_cast<Type*>(&result)[i] = static_cast<Type>(value + 0.5f);
		}

		return result;
	}

	template<PixelLike PixelType>
	struct WarpAffine : private Empty {
		__device__ __forceinline__ PixelType operator()(DataAccessor<const PixelType> input, int x, int y, Size size, const Matrix<float, 2, 3> invTransform) {
			const float srcX = invTransform.data[0][0] * x + invTransform.data[0][1] * y + invTransform.data[0][2];
			const float srcY = invTransform.data[1][0] * x + invTransform.data[1][1] * y + invTransform.data[1][2];

			const bool inputInRange = isInRange(srcX, 0.0f, size.width - 1.0f) && isInRange(srcY, 0.0f, size.height - 1.0f);
			return inputInRange ? bilinearInterpolate<PixelType>(input, size, srcX, srcY) : PixelType{};
		}
	};
}
