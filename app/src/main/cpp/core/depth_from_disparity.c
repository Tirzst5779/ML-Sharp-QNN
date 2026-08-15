/**
 * depth_from_disparity.c — 实现
 * depth_from_disparity.c — implementation
 */
#include "depth_from_disparity.h"
#include <math.h>

void depth_from_disparity(const float *disparity, float *depth,
                          int C, int H, int W, float d_factor,
                          float clamp_min, float clamp_max)
{
    int N = C * H * W;
    for (int i = 0; i < N; i++) {
        float d = disparity[i];
        if (d < clamp_min) d = clamp_min;
        if (d > clamp_max) d = clamp_max;
        depth[i] = d_factor / d;
    }
}