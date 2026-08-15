/**
 * initializer.h — 创建 Gaussian 基值和 feature_input
 * initializer.h — builds Gaussian base values and feature_input
 *
 *   基于 init_model (initializer.py:64-253)
 *   Based on init_model (initializer.py:64-253)
 *   全部是纯算术操作（无学习权重）
 *   Pure arithmetic only (no learned weights)
 */
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <stddef.h>

/* 基值结构体，对应 GaussianBaseValues */
/* Base values struct, matching GaussianBaseValues */
typedef struct {
    float *mean_x_ndc;          /* [1, 1, num_layers, H/2, W/2] */
    float *mean_y_ndc;          /* 同上 / same */
    float *mean_inverse_z_ndc;  /* 同上 (disparity layers) / same (disparity layers) */
    float *scales;             /* [1, 1, num_layers, H/2, W/2] */
    float *quaternions;        /* [1, 4, num_layers, H/2, W/2] */
    float *colors;             /* [1, 3, num_layers, H/2, W/2] */
    float *opacities;          /* [1, 1, num_layers, H/2, W/2] */
} GaussianBaseValues;

/* 初始器输出，对应 InitializerOutput */
/* Initializer output, matching InitializerOutput */
typedef struct {
    float *feature_input;     /* [1, 5, H, W] 即 prepare_feature_input(image, depth) 输出
                                 the output of prepare_feature_input(image, depth) */
    GaussianBaseValues base;  /* 基值（自己管理的内存）/ base values (self-managed memory) */
    float global_scale;       /* 缩放因子，scalar [1] / scale factor, scalar [1] */
} InitializerOutput;

/**
 * 运行 initializer 产生基值 + feature_input。
 * Runs the initializer to produce base values + feature_input.
 * @param image      [1, 3, H, W]  归一化 [0,1] 图像 / normalized image in [0,1]
 * @param depth      [1, 2, H, W]  metric depth
 * @param stride      下采样步长 (2) / downsample stride (2)
 * @param num_layers  层数 (2) / layer count (2)
 * @param base_depth  基深度 (10.0) / base depth (10.0)
 * @param scale_factor  (1.0)
 * @param init_disparity_factor  init_model 的 disparity_factor (1.0)
 * @param color_option  颜色初始化选项 (0=none, 1=first_layer, 2=all_layers)
 * @param normalize_depth  是否归一化深度 (true=1) / whether to normalize depth (true=1)
 * @return           InitializerOutput。调用方负责 free_output()
 *                   InitializerOutput. Caller is responsible for free_output().
 *
 * 注意: `image` [0,1], `depth` metric (m)，不是 gap/clamp 后的。
 * Note: `image` is in [0,1] and `depth` is metric (m), not gap/clamped values.
 */
InitializerOutput initializer_run(const float *image, const float *depth,
                                   int H, int W,
                                   int stride, int num_layers,
                                   float base_depth, float scale_factor,
                                   float init_disparity_factor,
                                   int color_option,
                                   int normalize_depth);

/* 释放 initializer 输出的内部内存 */
/* Frees the internal memory of an initializer output */
void initializer_free_output(InitializerOutput *out);

#ifdef __cplusplus
}
#endif

#endif /* INITIALIZER_H */