/**
 * unproject.h — NDC gaussians -> metric/world gaussians
 *
 *   apply_transform: mean @ R^T + t, 协方差 R @ diag(s)^2 @ R^T
 *   apply_transform: mean @ R^T + t, covariance R @ diag(s)^2 @ R^T
 *   当 extrinsics=identity 时 R 近似对角，可直接缩放。
 *   When extrinsics are identity, R is near-diagonal and a direct scale works.
 */
#ifndef UNPROJECT_H
#define UNPROJECT_H

#include "composer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 简化 unproject: 假设 extrinsics=identity, 退化为对角缩放。
 * Simplified unproject: assumes identity extrinsics, reduces to a diagonal scale.
 * @param gaussians  输入 NDC gaussians (平铺布局) / input NDC gaussians (flat layout)
 * @param f_px_x     缩放后 x 方向焦距 (像素, = f_px * new_w / orig_w)
 *                  x focal length after resize (px, = f_px * new_w / orig_w)
 * @param f_px_y     缩放后 y 方向焦距 (像素, = f_px * new_h / orig_h)
 *                  y focal length after resize (px, = f_px * new_h / orig_h)
 * @param img_w      缩放后图像宽 (1536) / resized image width (1536)
 * @param img_h      缩放后图像高 (1536) / resized image height (1536)
 * @param cx, cy     主点 (像素, 原始图像中心, 缩放不改变)
 *                  principal point (px, image center; unchanged by resize)
 * @param out        输出 metric gaussians (可复用输入指针) / output metric gaussians (input pointer may be reused)
 */
void unproject_simple(const Gaussians3DFlat *gaussians,
                       Gaussians3DFlat *out,
                       float f_px_x, float f_px_y, int img_w, int img_h,
                       float cx, float cy);

/**
 * 全量 SVD unproject (完整版, 支持非 identity extrinsics)。
 * Full SVD unproject (complete version, supports non-identity extrinsics).
 * @param gaussians  输入 NDC gaussians / input NDC gaussians
 * @param transform  3x4 仿射矩阵 (行优先, 12 个 float) / 3x4 affine matrix (row-major, 12 floats)
 * @param out        输出 metric gaussians。
 *                  Output metric gaussians.
 *                   out == gaussians 时原地变换 (逐点先读入局部变量再写同索引, 安全),
 *                  In-place transform when out == gaussians (safe: each point is read into
 *                  locals before writing the same index),
 *                   不分配新缓冲; out != gaussians 时内部 malloc mean/sv/quat 三份,
 *                  no new buffers are allocated; when out != gaussians it mallocs
 *                  mean/sv/quat internally,
 *                   colors/opacities 与输入共享指针, 调用方勿重复 free。
 *                  colors/opacities share the input pointers; the caller must not free them.
 */
void unproject_full(const Gaussians3DFlat *gaussians,
                     const float transform[12],
                     Gaussians3DFlat *out);

#ifdef __cplusplus
}
#endif

#endif /* UNPROJECT_H */