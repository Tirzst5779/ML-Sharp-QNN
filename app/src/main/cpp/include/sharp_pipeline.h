// sharp_pipeline.h — SHARP core 库化接口
// sharp_pipeline.h — library interface for the SHARP core
// 从 pipeline.c 的 main() 提取, 暴露 pre/merge/post 为可调用函数
// Extracted from pipeline.c's main(); exposes pre/merge/post as callable functions
// 供 JNI 层调用, 去掉命令行参数解析
// Called from the JNI layer; no command-line parsing
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 预处理: image.raw → 35×patch_p*.raw + x2.raw + input_list_*.txt
// Preprocess: image.raw -> 35×patch_p*.raw + x2.raw + input_list_*.txt
// imageRawPath: 输入 image.raw [1,3,1536,1536] NCHW f32
//               input image.raw [1,3,1536,1536] NCHW f32
// workDir: 工作目录 (输出文件放在此目录)
//          work directory (outputs are written here)
// 返回 0 成功, 非 0 失败
// Returns 0 on success, non-zero on failure
int sharp_pre(const char* imageRawPath, const char* workDir);

// Merge: HTP 输出 → 6 个合并特征 raw
// Merge: HTP outputs -> 6 merged feature raws
// workDir: 工作目录 / work directory
// peOutDir: patch_encoder 输出目录 (含 Result_0..34/)
//           patch_encoder output directory (contains Result_0..34/)
// ieOutDir: image_encoder 输出目录 (含 Result_0/)
//           image_encoder output directory (contains Result_0/)
// 返回 0 成功, 非 0 失败
// Returns 0 on success, non-zero on failure
int sharp_merge(const char* workDir, const char* peOutDir, const char* ieOutDir);

// Post: delta + disparity + image → output.ply
// Post: delta + disparity + image -> output.ply
// workDir: 工作目录 (含 image.raw, disparity.raw, delta.raw)
//          work directory (contains image.raw, disparity.raw, delta.raw)
// fpx: 焦距像素值 / focal length in pixels
// origW, origH: 原始图片宽高 / original image width and height
// outPlyPath: 输出 PLY 文件路径 / output PLY file path
// 返回 0 成功, 非 0 失败
// Returns 0 on success, non-zero on failure
int sharp_post(const char* workDir, float fpx, int origW, int origH, const char* outPlyPath);

// 预处理图片: 解码 JPEG/PNG → EXIF 旋转/焦距 → resize 1536 → image.raw
// Preprocess image: decode JPEG/PNG -> EXIF rotation/focal length -> resize 1536 -> image.raw
// imagePath: 输入图片路径 / input image path
// outRawPath: 输出 image.raw 路径 [1,3,1536,1536] NCHW f32 值域[0,1]
//             output image.raw path [1,3,1536,1536] NCHW f32 in [0,1]
// outFpx: 输出焦距像素值 / output focal length in pixels
// outDfactor: 输出 disparity_factor = f_px / orig_w
//             output disparity_factor = f_px / orig_w
// outOrigW, outOrigH: 输出原始图片宽高 (EXIF 旋转后)
//                     output original image size (after EXIF rotation)
// 返回 0 成功, 非 0 失败
// Returns 0 on success, non-zero on failure
int sharp_prep_image(const char* imagePath, const char* outRawPath,
                     float* outFpx, float* outDfactor, int* outOrigW, int* outOrigH);

// ============== 进度回调 ==============
// ============== Progress callback ==============
// 进度回调函数类型
// Progress callback function type
// stageId: 阶段 ID (见下方常量) / stage ID (see constants below)
// stageName: 阶段名称 / stage name
// current: 当前进度 / current progress
// total: 总数 / total
// elapsedMs: 已耗时(毫秒) / elapsed time (ms)
typedef void (*SharpProgressCallback)(int stageId, const char* stageName,
                                       int current, int total, long elapsedMs,
                                       const char* detail);

// 设置全局进度回调 (线程安全)
// Sets the global progress callback (thread-safe)
void sharp_set_progress_callback(SharpProgressCallback cb);

// 阶段 ID 常量 (与 Kotlin PipelineState.kt DEFAULT_STAGES 一一对应)
// Stage ID constants (one-to-one with Kotlin PipelineState.kt DEFAULT_STAGES)
#define SHARP_STAGE_PREP_IMAGE   0  // 解码图片 / image decode
#define SHARP_STAGE_PRE          1  // 预处理+切patch / preprocess + patch split
#define SHARP_STAGE_PE_INFER     2  // patch_encoder HTP 推理 / patch_encoder HTP inference
#define SHARP_STAGE_IE_INFER     3  // image_encoder HTP 推理 / image_encoder HTP inference
#define SHARP_STAGE_MERGE        4  // 合并特征 / feature merge
#define SHARP_STAGE_REST_A       5  // REST Seg A (特征融合) / REST Seg A (feature fusion)
#define SHARP_STAGE_REST_B       6  // REST Seg B (视差估计) / REST Seg B (disparity estimation)
#define SHARP_STAGE_REST_C       7  // REST Seg C (高斯增量) / REST Seg C (Gaussian deltas)
#define SHARP_STAGE_POST         8  // 后处理 (点云生成) / postprocess (point cloud)

#ifdef __cplusplus
}
#endif