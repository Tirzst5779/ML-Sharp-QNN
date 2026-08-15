/**
 * save_ply.c — 写入 .ply 文件
 * save_ply.c — writes .ply files
 */
#include "save_ply.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __ANDROID__
#include <android/log.h>
#define PLY_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "SharpCore", __VA_ARGS__)
#else
#define PLY_LOGE(...) fprintf(stderr, "[SharpCore] " __VA_ARGS__)
#endif

static float linear_to_srgb(float c)
{
    if (c <= 0.0031308f)
        return c * 12.92f;
    return 1.055f * powf(c, 1.0f/2.4f) - 0.055f;
}

static float rgb_to_sh0(float c)
{
    /* convert_rgb_to_spherical_harmonics: (rgb - 0.5) / sqrt(1/(4*pi)) */
    static const float inv_coeff = 1.0f / 0.28209479177f; /* sqrt(1/(4*pi)) ≈ 0.282095 */
    return (c - 0.5f) * inv_coeff;
}

static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int save_ply(const Gaussians3DFlat *g, float f_px,
             int image_w, int image_h, const char *fpath)
{
    FILE *fp = fopen(fpath, "wb");
    if (!fp) { PLY_LOGE("Cannot open %s\n", fpath); return 1; }

    int N = g->num_points;

    /* PLY header */
    fprintf(fp, "ply\nformat binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", N);
    fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
    fprintf(fp, "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n");
    fprintf(fp, "property float opacity\n");
    fprintf(fp, "property float scale_0\nproperty float scale_1\nproperty float scale_2\n");
    fprintf(fp, "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n");
    fprintf(fp, "element extrinsic 16\nproperty float extrinsic\n");
    fprintf(fp, "element intrinsic 9\nproperty float intrinsic\n");
    fprintf(fp, "element image_size 2\nproperty uint image_size\n");
    fprintf(fp, "element frame 2\nproperty int frame\n");
    fprintf(fp, "element disparity 2\nproperty float disparity\n");
    fprintf(fp, "element color_space 1\nproperty uchar color_space\n");
    fprintf(fp, "element version 3\nproperty uchar version\n");
    fprintf(fp, "end_header\n");

    /* vertex 数据: xyz, f_dc(线性RGB->sRGB->SH0), opacity(logits), scale(log), rot */
    /* Vertex data: xyz, f_dc (linearRGB->sRGB->SH0), opacity (logits), scale (log), rot */
    for (int i = 0; i < N; i++) {
        float x = g->mean_vectors[0*N+i];
        float y = g->mean_vectors[1*N+i];
        float z = g->mean_vectors[2*N+i];

        /* 颜色: SHARP 预测 linearRGB, 导出时强制转 sRGB */
        /* Color: SHARP predicts linearRGB, converted to sRGB on export */
        float sr = linear_to_srgb(g->colors[0*N+i]);
        float sg = linear_to_srgb(g->colors[1*N+i]);
        float sb = linear_to_srgb(g->colors[2*N+i]);
        float sh0_r = rgb_to_sh0(sr);
        float sh0_g = rgb_to_sh0(sg);
        float sh0_b = rgb_to_sh0(sb);

        /* opacity logits = log(t/(1-t)) */
        float op = g->opacities[i];
        float logit_op = logf(op / (1.0f - op));

        /* scale logits = log(singular_values) */
        float log_sv0 = logf(g->singular_values[0*N+i]);
        float log_sv1 = logf(g->singular_values[1*N+i]);
        float log_sv2 = logf(g->singular_values[2*N+i]);

        float qw = g->quaternions[0*N+i];
        float qx = g->quaternions[1*N+i];
        float qy = g->quaternions[2*N+i];
        float qz = g->quaternions[3*N+i];

        fwrite(&x, sizeof(float), 1, fp);
        fwrite(&y, sizeof(float), 1, fp);
        fwrite(&z, sizeof(float), 1, fp);
        fwrite(&sh0_r, sizeof(float), 1, fp);
        fwrite(&sh0_g, sizeof(float), 1, fp);
        fwrite(&sh0_b, sizeof(float), 1, fp);
        fwrite(&logit_op, sizeof(float), 1, fp);
        fwrite(&log_sv0, sizeof(float), 1, fp);
        fwrite(&log_sv1, sizeof(float), 1, fp);
        fwrite(&log_sv2, sizeof(float), 1, fp);
        fwrite(&qw, sizeof(float), 1, fp);
        fwrite(&qx, sizeof(float), 1, fp);
        fwrite(&qy, sizeof(float), 1, fp);
        fwrite(&qz, sizeof(float), 1, fp);
    }

    /* extrinsic: eye(4) */
    float eye4[16];
    memset(eye4, 0, sizeof(eye4));
    eye4[0] = eye4[5] = eye4[10] = eye4[15] = 1.0f;
    fwrite(eye4, sizeof(float), 16, fp);

    /* intrinsic: [f,0,W/2, 0,f,H/2, 0,0,1] */
    float intr[9] = {f_px, 0.0f, image_w * 0.5f,
                     0.0f, f_px, image_h * 0.5f,
                     0.0f, 0.0f, 1.0f};
    fwrite(intr, sizeof(float), 9, fp);

    /* image_size: [W, H] (u4) */
    unsigned int img_size[2] = { (unsigned int)image_w, (unsigned int)image_h };
    fwrite(img_size, sizeof(unsigned int), 2, fp);

    /* frame: [1, N] (i4) */
    int frames[2] = { 1, N };
    fwrite(frames, sizeof(int), 2, fp);

    /* disparity: quantile 0.1/0.9 of 1/mean_z (线性插值, 同 torch.quantile) */
    /* disparity: 0.1/0.9 quantiles of 1/mean_z (linear interpolation, like torch.quantile) */
    {
        float *disp = (float*)malloc((size_t)N * sizeof(float));
        if (!disp) { fclose(fp); return 1; }
        for (int i = 0; i < N; i++)
            disp[i] = 1.0f / g->mean_vectors[2*N+i];
        qsort(disp, (size_t)N, sizeof(float), cmp_float);
        float qs[2] = { 0.1f, 0.9f };
        float quant[2];
        for (int k = 0; k < 2; k++) {
            float pos = qs[k] * (N - 1);
            int lo = (int)pos;
            int hi = lo + 1 < N ? lo + 1 : lo;
            float frac = pos - lo;
            quant[k] = disp[lo] * (1.0f - frac) + disp[hi] * frac;
        }
        fwrite(quant, sizeof(float), 2, fp);
        free(disp);
    }

    /* color_space: 0 = sRGB (u1) */
    unsigned char cs = 0;
    fwrite(&cs, sizeof(unsigned char), 1, fp);

    /* version: [1,5,0] (u1) */
    unsigned char ver[3] = { 1, 5, 0 };
    fwrite(ver, sizeof(unsigned char), 3, fp);

    fclose(fp);
    return 0;
}