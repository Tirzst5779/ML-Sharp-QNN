/**
 * composer.h — GaussianComposer: delta + base_values -> NDC gaussians
 *
 *   参考 composer.py:92-251
 *   Reference: composer.py:92-251
 *   全部纯算术 + 激活函数
 *   Pure arithmetic + activation functions only
 */
#ifndef COMPOSER_H
#define COMPOSER_H

#include "initializer.h"  /* GaussianBaseValues */

#ifdef __cplusplus
extern "C" {
#endif

/* 输出 3D Gaussians (平铺布局) */
/* Output 3D Gaussians (flat layout) */
typedef struct {
    float *mean_vectors;          /* [B, L*H*W, 3] */
    float *singular_values;       /* [B, L*H*W, 3] */
    float *quaternions;           /* [B, L*H*W, 4] */
    float *colors;                /* [B, L*H*W, 3] */
    float *opacities;             /* [B, L*H*W] */
    int num_points;
} Gaussians3DFlat;

/* 参数 */
/* Params */
typedef struct {
    float delta_xy;       /* DeltaFactor.xy   (0.001) */
    float delta_z;        /* DeltaFactor.z    (0.001) */
    float delta_color;    /* DeltaFactor.color (0.1) */
    float delta_opacity;  /* DeltaFactor.opacity (1.0) */
    float delta_scale;    /* DeltaFactor.scale (1.0) */
    float delta_quat;     /* DeltaFactor.quaternion (1.0) */
    float min_scale;      /* (0.0) */
    float max_scale;      /* (10.0) */
    int color_activation; /* 0=sigmoid, 1=softplus, 2=exp */
    int color_space;      /* 0=sRGB, 1=linearRGB */
    int base_scale_on_predicted_mean; /* (1=true) */
} ComposerParams;

/**
 * 运行 composer, 从 delta + base_values 生成 NDC gaussians。
 * Runs the composer, building NDC gaussians from delta + base_values.
 * @param delta     [1, 14, L, H, W]  (来自 prediction_head 输出 / from the prediction_head output)
 * @param base      基值 (来自 initializer) / base values (from the initializer)
 * @param global_scale  缩放因子 (来自 initializer) / scale factor (from the initializer)
 * @param H,W       delta 的空间尺寸 (768) / delta spatial size (768)
 * @param L         num_layers (2)
 * @param params    参数 / params
 * @param flatten_output  是否平铺输出 (1=是) / whether to flatten the output (1=yes)
 * @return          平铺的 Gaussians3D / the flattened Gaussians3D
 */
Gaussians3DFlat composer_run(
    const float *delta,
    const GaussianBaseValues *base,
    float global_scale,
    int H, int W, int L,
    const ComposerParams *params,
    int flatten_output);

/* 释放 composer 输出内存 */
/* Frees the composer output memory */
void composer_free_output(Gaussians3DFlat *out);

#ifdef __cplusplus
}
#endif

#endif /* COMPOSER_H */