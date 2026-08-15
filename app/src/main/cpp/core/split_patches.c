/**
 * split_patches.c --- 滑窗切割实现
 * split_patches.c --- sliding-window split implementation
 */
#include "split_patches.h"
#include <string.h>
#include <math.h>

int split_single(const float *src, float *dst,
                 int C, int H, int W,
                 int patch_size, float overlap_ratio)
{
    int patch_stride = (int)(patch_size * (1.0f - overlap_ratio));
    int steps = (int)ceil((double)(H - patch_size) / (double)patch_stride) + 1;
    int patch_pixels = patch_size * patch_size;
    int idx = 0;

    for (int j = 0; j < steps; j++) {
        int j0 = j * patch_stride;
        int j1 = j0 + patch_size;

        for (int i = 0; i < steps; i++) {
            int i0 = i * patch_stride;
            int i1 = i0 + patch_size;

            float *patch_out = dst + idx * C * patch_pixels;

            for (int c = 0; c < C; c++) {
                const float *src_c = src + c * H * W;
                float       *dst_c = patch_out + c * patch_pixels;

                for (int y = j0; y < j1; y++) {
                    const float *src_row = src_c + y * W + i0;
                    float       *dst_row = dst_c + (y - j0) * patch_size;
                    memcpy(dst_row, src_row, patch_size * sizeof(float));
                }
            }
            idx++;
        }
    }
    return idx;
}

void split_pyramid_batch(const float *x0_batch, const float *x1_batch,
                         const float *x2_batch, float *patches,
                         int batch, int C,
                         int H_hi, int W_hi,
                         int H_mid, int W_mid,
                         int H_lo, int W_lo,
                         int patch_size)
{
    int plane_x0 = C * H_hi * W_hi;
    int plane_x1 = C * H_mid * W_mid;
    int plane_x2 = C * H_lo * W_lo;
    int patch_plane = C * patch_size * patch_size;

    for (int b = 0; b < batch; b++) {
        float *bp = patches + b * 35 * patch_plane;

        int n0 = split_single(x0_batch + b * plane_x0,
                              bp,
                              C, H_hi, W_hi, patch_size, 0.25f);

        int n1 = split_single(x1_batch + b * plane_x1,
                              bp + n0 * patch_plane,
                              C, H_mid, W_mid, patch_size, 0.5f);

        const float *x2_src = x2_batch + b * plane_x2;
        memcpy(bp + (n0 + n1) * patch_plane, x2_src, plane_x2 * sizeof(float));
    }
}