/**
 * normalizer.c — AffineRangeNormalizer 实现
 * normalizer.c — AffineRangeNormalizer implementation
 */
#include "normalizer.h"

void normalizer_apply(const float *src, float *dst,
                      int batch, int channel, int height, int width,
                      float scale, float bias)
{
    const int N = batch * channel * height * width;
    for (int i = 0; i < N; i++) {
        dst[i] = src[i] * scale + bias;
    }
}

void normalizer_apply_inplace(float *data,
                              int batch, int channel, int height, int width,
                              float scale, float bias)
{
    normalizer_apply(data, data, batch, channel, height, width, scale, bias);
}