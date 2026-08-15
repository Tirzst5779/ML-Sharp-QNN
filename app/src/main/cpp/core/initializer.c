/**
 * initializer.c — Gaussian 基值创建
 * initializer.c — Gaussian base value creation
 *   initializer.py:64-253 的 C 实现
 *   C implementation of initializer.py:64-253
 *   纯算术，无学习权重
 *   Pure arithmetic, no learned weights
 */
#include "initializer.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>

/* 2x2 max_pool stride=2, CHW */
static void max_pool2d_2x2(const float *src, float *dst, int C, int H, int W)
{
    int ho = H/2, wo = W/2;
    for (int c = 0; c < C; c++) {
        const float *s = src + c * H * W;
        float *d = dst + c * ho * wo;
        for (int i = 0; i < ho; i++) {
            for (int j = 0; j < wo; j++) {
                float v0 = s[(2*i)*W + 2*j];
                float v1 = s[(2*i)*W + 2*j+1];
                float v2 = s[(2*i+1)*W + 2*j];
                float v3 = s[(2*i+1)*W + 2*j+1];
                float mx = v0;
                if (v1 > mx) mx = v1;
                if (v2 > mx) mx = v2;
                if (v3 > mx) mx = v3;
                d[i*wo + j] = mx;
            }
        }
    }
}

/* 2x2 avg_pool stride=2 */
static void avg_pool2d_2x2(const float *src, float *dst, int C, int H, int W)
{
    int ho = H/2, wo = W/2;
    for (int c = 0; c < C; c++) {
        const float *s = src + c * H * W;
        float *d = dst + c * ho * wo;
        for (int i = 0; i < ho; i++) {
            for (int j = 0; j < wo; j++) {
                float sum = s[(2*i)*W + 2*j] + s[(2*i)*W + 2*j+1]
                          + s[(2*i+1)*W + 2*j] + s[(2*i+1)*W + 2*j+1];
                d[i*wo + j] = sum * 0.25f;
            }
        }
    }
}

/* rescale_depth: 深度归一化使最小值 ~1.0 */
/* rescale_depth: rescales depth so the minimum becomes ~1.0 */
static float rescale_depth(float *depth, int C, int H, int W,
                           float depth_min, float depth_max)
{
    double mn = DBL_MAX;
    int N = C * H * W;
    for (int i = 0; i < N; i++)
        if (depth[i] < mn) mn = depth[i];
    float factor = depth_min / ((float)mn + 1e-6f);
    for (int i = 0; i < N; i++) {
        float v = depth[i] * factor;
        if (v > depth_max) v = depth_max;
        depth[i] = v;
    }
    return factor;
}

/* prepare_feature_input: [1,5,H,W] */
static void prepare_feat_input(const float *img, const float *dep,
                                float *out, int H, int W, float idf)
{
    int N = H * W;
    for (int i = 0; i < N; i++) {
        out[0*N+i] = 2.0f*img[0*N+i] - 1.0f;
        out[1*N+i] = 2.0f*img[1*N+i] - 1.0f;
        out[2*N+i] = 2.0f*img[2*N+i] - 1.0f;
        float nd0 = idf / dep[i];
        out[3*N+i] = 2.0f*nd0 - 1.0f;
        float nd1 = idf / dep[N+i];
        out[4*N+i] = 2.0f*nd1 - 1.0f;
    }
}

/* create_base_xy: meshgrid, [1,1,L,bh,bw] */
static void create_base_xy(float *bx, float *by,
                            int H, int W, int S, int L)
{
    int bh = H/S, bw = W/S;
    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            float fx = (2.0f*(x*S + 0.5f*S)/W) - 1.0f;
            float fy = (2.0f*(y*S + 0.5f*S)/H) - 1.0f;
            for (int l = 0; l < L; l++) {
                bx[l*bh*bw + y*bw + x] = fx;
                by[l*bh*bw + y*bw + x] = fy;
            }
        }
    }
}

/* create_surface_layer: max_pool2d(1/depth, 2, 2), [1,1,bh,bw] */
static void create_surf_layer(const float *dep_ch, float *out,
                               int H, int W, int S)
{
    int bh = H/S, bw = W/S;
    float *inv = (float*)malloc(H*W*sizeof(float));
    for (int i = 0; i < H*W; i++)
        inv[i] = 1.0f / dep_ch[i];
    max_pool2d_2x2(inv, out, 1, H, W);
    free(inv);
}

/* ── 主入口 ── */
/* ── Main entry ── */
InitializerOutput initializer_run(
    const float *image, const float *depth,
    int H, int W,
    int stride, int num_layers,
    float base_depth, float scale_factor,
    float init_disparity_factor,
    int color_option,
    int normalize_depth)
{
    int bh = H/stride, bw = W/stride;
    int nb = num_layers * bh * bw;

    InitializerOutput out;
    memset(&out, 0, sizeof(out));

    /* 拷贝 depth（rescale_depth 会修改） */
    /* Copy depth (rescale_depth modifies it) */
    float *dtmp = (float*)malloc(2*H*W*sizeof(float));
    memcpy(dtmp, depth, 2*H*W*sizeof(float));

    float gs = 0.0f;
    if (normalize_depth) {
        float f = rescale_depth(dtmp, 2, H, W, 1.0f, 100.0f);
        gs = 1.0f / f;
    }
    out.global_scale = gs;

    /* 分配基值内存 */
    /* Allocate base value memory */
    out.base.mean_x_ndc         = (float*)calloc(nb, sizeof(float));
    out.base.mean_y_ndc         = (float*)calloc(nb, sizeof(float));
    out.base.mean_inverse_z_ndc = (float*)calloc(nb, sizeof(float));
    out.base.scales             = (float*)calloc(nb, sizeof(float));
    out.base.quaternions        = (float*)calloc(4*nb, sizeof(float));
    out.base.colors             = (float*)calloc(3*nb, sizeof(float));
    out.base.opacities          = (float*)calloc(nb, sizeof(float));

    /* base_xy */
    create_base_xy(out.base.mean_x_ndc, out.base.mean_y_ndc, H, W, stride, num_layers);

    /* disparity layers */
    float *disp = (float*)calloc(nb, sizeof(float));
    float *ch0 = (float*)malloc(nb*sizeof(float));
    float *ch1 = (float*)malloc(nb*sizeof(float));
    create_surf_layer(dtmp, ch0, H, W, stride);
    create_surf_layer(dtmp + H*W, ch1, H, W, stride);
    memcpy(disp, ch0, nb*sizeof(float));  /* L=2: first layer = ch0 */
    for (int i = 0; i < nb; i++) {
        if (i < bh*bw)
            disp[i] = ch0[i];
        else
            disp[i] = ch1[i - bh*bw];
    }
    memcpy(out.base.mean_inverse_z_ndc, disp, nb*sizeof(float));

    /* scales = 1/disparity * s2 * len(W) 原始公式 */
    /* scales = 1/disparity * s2 * len(W), the original formula */
    float ds = scale_factor * 2.0f * stride / (float)W;
    for (int i = 0; i < nb; i++) {
        float inv = disp[i] > 1e-8f ? 1.0f/disp[i] : 1e8f;
        out.base.scales[i] = inv * ds;
    }

    /* quaternions */
    for (int i = 0; i < nb; i++) {
        out.base.quaternions[0*nb+i] = 1.0f;
        out.base.quaternions[1*nb+i] = 0.0f;
        out.base.quaternions[2*nb+i] = 0.0f;
        out.base.quaternions[3*nb+i] = 0.0f;
    }

    /* opacities */
    float op = (1.0f/num_layers < 0.5f) ? (1.0f/num_layers) : 0.5f;
    for (int i = 0; i < nb; i++) out.base.opacities[i] = op;

    /* colors = 0.5 + avg_pool2d, 布局 [3, num_layers, bh, bw] (与 torch 一致) */
    /* colors = 0.5 + avg_pool2d, layout [3, num_layers, bh, bw] (torch-consistent) */
    for (int i = 0; i < 3*nb; i++) out.base.colors[i] = 0.5f;
    if (color_option == 2 || color_option == 1) {
        float *pl = (float*)malloc(3*bh*bw*sizeof(float));
        avg_pool2d_2x2(image, pl, 3, H, W);
        for (int c = 0; c < 3; c++) {
            for (int l = 0; l < num_layers; l++) {
                for (int i = 0; i < bh*bw; i++) {
                    out.base.colors[c*nb + l*bh*bw + i] = pl[c*bh*bw + i];
                }
            }
        }
        free(pl);
    }

    /* feature_input */
    out.feature_input = (float*)malloc(5*H*W*sizeof(float));
    prepare_feat_input(image, dtmp, out.feature_input, H, W, init_disparity_factor);

    free(dtmp); free(disp); free(ch0); free(ch1);
    return out;
}

void initializer_free_output(InitializerOutput *out)
{
    if (!out) return;
    free(out->base.mean_x_ndc);
    free(out->base.mean_y_ndc);
    free(out->base.mean_inverse_z_ndc);
    free(out->base.scales);
    free(out->base.quaternions);
    free(out->base.colors);
    free(out->base.opacities);
    free(out->feature_input);
    memset(&out->base, 0, sizeof(out->base));
    out->feature_input = NULL;
}