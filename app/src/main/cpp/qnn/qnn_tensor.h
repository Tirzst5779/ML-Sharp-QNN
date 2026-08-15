// qnn_tensor.h — QNN 张量工具
// qnn_tensor.h — QNN tensor utilities
// float32 ↔ UFIXED_POINT_16/8 量化转换, 张量内存管理
// float32 <-> UFIXED_POINT_16/8 quantized conversion, tensor memory helpers
#pragma once

#include <cstdint>
#include <vector>

namespace qnn {

// float → UFIXED_POINT_16 量化
// float -> UFIXED_POINT_16 quantization
// 公式: quantized = clamp(round(float / scale) - offset, 0, 65535)
// Formula: quantized = clamp(round(float / scale) - offset, 0, 65535)
void floatToUfixed16(const float* src, uint16_t* dst, size_t count, float scale, int32_t offset);

// float → UFIXED_POINT_8 量化
// float -> UFIXED_POINT_8 quantization
void floatToUfixed8(const float* src, uint8_t* dst, size_t count, float scale, int32_t offset);

// UFIXED_POINT_16 → float 反量化
// UFIXED_POINT_16 -> float dequantization
// 公式: float = (quantized + offset) * scale
// Formula: float = (quantized + offset) * scale
// (offset 是零点偏移, 通常为负数, quantized=32768 + offset=-32768 → float=0)
// (offset is the zero-point shift, usually negative; quantized=32768 + offset=-32768 -> float=0)
void ufixed16ToFloat(const uint16_t* src, float* dst, size_t count, float scale, int32_t offset);

// UFIXED_POINT_8 → float 反量化
// UFIXED_POINT_8 -> float dequantization
// 公式: float = (quantized + offset) * scale
// Formula: float = (quantized + offset) * scale
void ufixed8ToFloat(const uint8_t* src, float* dst, size_t count, float scale, int32_t offset);

// 计算张量元素数 (dims 乘积)
// Element count of a tensor (product of dims)
size_t calculateElementCount(const std::vector<uint32_t>& dims);

// NCHW → NHWC 转置 (float32, 4D)
// NCHW -> NHWC transpose (float32, 4D)
void nchwToNhwc(const float* src, float* dst, int n, int c, int h, int w);

// NHWC → NCHW 转置 (float32, 4D)
// NHWC -> NCHW transpose (float32, 4D)
void nhwcToNchw(const float* src, float* dst, int n, int c, int h, int w);

} // namespace qnn