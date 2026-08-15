// sharp_jni.cpp — JNI 桥接层
// sharp_jni.cpp — JNI bridge layer
// 连接 Kotlin (com.sharp.qnn.pipeline.QnnJni) 与 native C++ 层
// Connects Kotlin (com.sharp.qnn.pipeline.QnnJni) with the native C++ layer
// 实现: QNN HTP 运行时管理 + SHARP pipeline 调度 + 进度回调
// Implements: QNN HTP runtime management + SHARP pipeline orchestration + progress callbacks
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstring>
#include <memory>
#include <map>
#include <algorithm>
#include <sys/stat.h>  // mkdir
#include <dlfcn.h>     // probeHtpArch: dlopen/dlsym
#include <pthread.h>   // pthread_key_t: balances JNI DetachCurrentThread
#include <mutex>       // guards init/destroy/compile lifecycle

// SHARP core 接口
// SHARP core interface
#include "sharp_pipeline.h"
// QNN HTP 集成
// QNN HTP integration
#include "qnn_runtime.h"
#include "qnn_dlc_compiler.h"
#include "qnn_tensor.h"

#include "QnnDevice.h"
#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"

#define TAG "SHARP_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

// ============== 全局状态 ==============
// ============== Global state ==============
static JavaVM* g_jvm = nullptr;

// 互斥锁: 保护初始化/销毁/编译生命周期
// Mutex: guards the init/destroy/compile lifecycle
static std::mutex g_initMutex;

// QNN 共享状态: 单 backend + 单 device + 单 log, 五个 runtime 共享
// QNN shared state: one backend + one device + one log, shared by the five runtimes
// (官方标准结构; 最后一个 runtime 释放时销毁)
// (the official layout; destroyed when the last runtime releases it)
static qnn::QnnSharedState g_qnnShared;

// 五个 QNN 运行时: PE, IE, REST_A/B/C
// Five QNN runtimes: PE, IE, REST_A/B/C
static std::unique_ptr<qnn::HtpRuntime> g_peRuntime;
static std::unique_ptr<qnn::HtpRuntime> g_ieRuntime;
static std::unique_ptr<qnn::HtpRuntime> g_restARuntime;
static std::unique_ptr<qnn::HtpRuntime> g_restBRuntime;
static std::unique_ptr<qnn::HtpRuntime> g_restCRuntime;
static std::unique_ptr<qnn::DlcCompiler> g_dlcCompiler;
static bool g_qnnInitialized = false;

// HTP 性能配置: 锁角模式 (默认, 锁 MAX) / 自动调角模式 (区间 + DCVS 模式)
// HTP performance config: locked-corner mode (default, locked MAX) / adaptive mode (range + DCVS mode)
// 仅在 nativeInit (创建共享 backend/device) 时生效, 更改后需重启进程
// Takes effect only in nativeInit (when the shared backend/device is created); changes require a process restart
static qnn::PerfConfig g_perfCfg;

// REST 段间内存缓存: 避免段 A→B→C 之间的文件 IO
// Inter-segment memory cache for REST: avoids file IO between segments A->B->C
// 键 = 张量名 (QNN 模型中的实际名称), 值 = float 数据
// Key = tensor name (the actual name in the QNN model), value = float data
static std::map<std::string, std::vector<float>> g_restCache;
// 缓存总量: 由设备可用内存自动决定 (nativeInit 时计算), 默认 8MB/16MB
// Cache size: determined automatically from available device memory (computed at nativeInit), default 8MB/16MB
// 单张量阈值: 仅小张量进缓存 (大边张量 37.7MB / disparity 18.9MB 走文件)
// Per-tensor threshold: only small tensors enter the cache (the big edge tensor 37.7MB / disparity 18.9MB go to files)
static size_t g_restCacheMaxBytes = 8 * 1024 * 1024;
static size_t g_restCacheTotalBudget = 16 * 1024 * 1024;
static size_t g_restCacheBytes = 0;

// 尝试将张量加入缓存: 单张量 < 阈值且总量不超预算才缓存;
// Tries to cache a tensor: only if it is below the per-tensor threshold and the total stays within budget;
// 加入后超过预算则清空整个缓存 (宁可全部走文件, 避免峰值内存失控)
// on budget overflow the whole cache is cleared (prefer files over an uncontrolled memory peak)
// data 为量化原始数据 (uint16/uint8), 小张量反量化后缓存为 float
// data holds raw quantized values (uint16/uint8); small tensors are dequantized and cached as float
static void restCachePut(const std::string& name, const void* raw,
                         size_t count, const qnn::HtpRuntime::TensorInfo* oinfo) {
    size_t bytes = count * sizeof(float);
    if (bytes >= g_restCacheMaxBytes) return;
    if (g_restCacheBytes + bytes > g_restCacheTotalBudget) {
        g_restCache.clear();
        g_restCacheBytes = 0;
    }
    std::vector<float> fdata(count);
    if (oinfo && oinfo->dataType == QNN_DATATYPE_UFIXED_POINT_16) {
        qnn::ufixed16ToFloat(reinterpret_cast<const uint16_t*>(raw), fdata.data(),
                             count, oinfo->scale, oinfo->offset);
    } else if (oinfo && oinfo->dataType == QNN_DATATYPE_UFIXED_POINT_8) {
        qnn::ufixed8ToFloat(reinterpret_cast<const uint8_t*>(raw), fdata.data(),
                            count, oinfo->scale, oinfo->offset);
    } else {
        std::memcpy(fdata.data(), raw, count * sizeof(float));
    }
    g_restCache[name] = std::move(fdata);
    g_restCacheBytes += bytes;
}

// 进度回调 (Kotlin 侧 ProgressCallback 对象的全局引用)
// Progress callbacks (global refs to the Kotlin-side ProgressCallback object)
static jobject g_callbackObj = nullptr;
static jmethodID g_onStageStart = nullptr;
static jmethodID g_onProgress = nullptr;
static jmethodID g_onStageComplete = nullptr;
static jmethodID g_onLog = nullptr;
static jmethodID g_onError = nullptr;

// ============== JNI 回调工具 ==============
// ============== JNI callback helpers ==============

// current thread 退出时自动 DetachCurrentThread (配平 AttachCurrentThread)
// Detaches the current thread automatically on exit (balancing AttachCurrentThread)
// 经 pthread_key 析构函数实现, 避免原生线程泄漏 JVM 引用 (官方 JNI 规范要求)
// via a pthread_key destructor, so native threads never leak JVM refs (as required by the JNI spec)
static pthread_key_t g_jniTlsKey;
static pthread_once_t g_jniTlsKeyOnce = PTHREAD_ONCE_INIT;

static void jniTlsDestructor(void* /*env*/) {
    if (g_jvm) {
        g_jvm->DetachCurrentThread();
    }
}

static void makeJniTlsKey() {
    pthread_key_create(&g_jniTlsKey, jniTlsDestructor);
}

// 确保 current thread attached to JVM
// Ensures the current thread is attached to the JVM
static JNIEnv* getJniEnv() {
    if (!g_jvm) return nullptr;
    JNIEnv* env = nullptr;
    g_jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (env) return env;
    // 本线程尚未 attach: 首次 attach 并记录到 TLS, 线程退出时自动 detach
    // Not attached yet: attach once, record the env in TLS, and detach automatically on thread exit
    pthread_once(&g_jniTlsKeyOnce, makeJniTlsKey);
    env = static_cast<JNIEnv*>(pthread_getspecific(g_jniTlsKey));
    if (env) return env;
    JavaVMAttachArgs args = {JNI_VERSION_1_6, "SHARPNative", nullptr};
    if (JNI_OK == g_jvm->AttachCurrentThread(&env, &args) && env) {
        pthread_setspecific(g_jniTlsKey, env);
    }
    return env;
}

// 前向声明: JNI 异常检查 (定义在下方工具函数区)
// Forward declaration: JNI exception check (defined below in the utilities section)
static bool checkJniException(JNIEnv* env);

// 回调: 阶段开始
// Callback: stage start
static void cbStageStart(int stageId, const char* stageName) {
    JNIEnv* env = getJniEnv();
    if (!env || !g_callbackObj) return;
    jstring jname = env->NewStringUTF(stageName);
    if (checkJniException(env) || !jname) return;
    env->CallVoidMethod(g_callbackObj, g_onStageStart, stageId, jname);
    env->DeleteLocalRef(jname);
}

// 回调: 进度更新
// Callback: progress update
static void cbProgress(int stageId, int current, int total, long elapsedMs, const char* detail) {
    JNIEnv* env = getJniEnv();
    if (!env || !g_callbackObj) return;
    jstring jdetail = env->NewStringUTF(detail ? detail : "");
    if (checkJniException(env) || !jdetail) return;
    env->CallVoidMethod(g_callbackObj, g_onProgress, stageId, current, total, (jlong)elapsedMs, jdetail);
    env->DeleteLocalRef(jdetail);
}

// 回调: 阶段完成
// Callback: stage complete
static void cbStageComplete(int stageId, const char* stageName, long elapsedMs) {
    JNIEnv* env = getJniEnv();
    if (!env || !g_callbackObj) return;
    jstring jname = env->NewStringUTF(stageName);
    if (checkJniException(env) || !jname) return;
    env->CallVoidMethod(g_callbackObj, g_onStageComplete, stageId, jname, (jlong)elapsedMs);
    env->DeleteLocalRef(jname);
}

// 回调: 日志
// Callback: log
static void cbLog(int level, const char* message) {
    JNIEnv* env = getJniEnv();
    if (!env || !g_callbackObj) return;
    jstring jmsg = env->NewStringUTF(message);
    if (checkJniException(env) || !jmsg) return;
    env->CallVoidMethod(g_callbackObj, g_onLog, level, jmsg);
    env->DeleteLocalRef(jmsg);
}

// 回调: 错误
// Callback: error
static void cbError(int stageId, const char* message) {
    JNIEnv* env = getJniEnv();
    if (!env || !g_callbackObj) return;
    jstring jmsg = env->NewStringUTF(message);
    if (checkJniException(env) || !jmsg) return;
    env->CallVoidMethod(g_callbackObj, g_onError, stageId, jmsg);
    env->DeleteLocalRef(jmsg);
}

// ============== SHARP core 进度回调 (C 接口) ==============
// ============== SHARP core progress callback (C interface) ==============
// 由 sharp_pipeline.c 内部调用, 转发到 Kotlin
// Invoked internally by sharp_pipeline.c, forwarded to Kotlin
static void sharpProgressCallback(int stageId, const char* stageName,
                                   int current, int total, long elapsedMs,
                                   const char* detail) {
    cbProgress(stageId, current, total, elapsedMs, detail);
}

// ============== 工具函数 ==============
// ============== Utilities ==============

// 读取 .raw 文件到 float 数组
// Reads a .raw file into a float array
static std::vector<float> readRawFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { LOGE("cannot open: %s", path.c_str()); return {}; }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// 写 float 数组到 .raw 文件
// Writes a float array to a .raw file
static bool writeRawFile(const std::string& path, const float* data, size_t count) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { LOGE("cannot write: %s", path.c_str()); return false; }
    f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    return true;
}

// 递归创建目录 (类似 mkdir -p)
// Creates directories recursively (like mkdir -p)
static void ensureDir(const std::string& path) {
    if (path.empty()) return;
    std::string current;
    current.reserve(path.size());
    for (size_t i = 0; i < path.size(); i++) {
        char c = path[i];
        current += c;
        if (c == '/' && current.size() > 1) {
            mkdir(current.c_str(), 0755);
        }
    }
    mkdir(path.c_str(), 0755);
}

// HTP 架构字符串 → 枚举
// HTP architecture string -> enum
static qnn::HtpArch parseArch(const std::string& arch) {
    if (arch == "V68") return qnn::HtpArch::V68;
    if (arch == "V69") return qnn::HtpArch::V69;
    if (arch == "V73") return qnn::HtpArch::V73;
    if (arch == "V75") return qnn::HtpArch::V75;
    if (arch == "V81") return qnn::HtpArch::V81;
    return qnn::HtpArch::V79; // 默认 V79 (Snapdragon 8 Elite) / default V79 (Snapdragon 8 Elite)
}

// 探测当前设备实际的 HTP 架构 (官方做法: QnnDevice_getPlatformInfo)
// Probes the device's actual HTP architecture (official approach: QnnDevice_getPlatformInfo)
// 返回枚举值字符串 ("V68".."V89"); 失败返回空串
// Returns the enum-value string ("V68".."V89"); returns an empty string on failure
static std::string probeHtpArch(const std::string& libDir) {
    void* htpLib = dlopen((libDir + "/libQnnHtp.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!htpLib) {
        LOGE("probeHtpArch: dlopen libQnnHtp.so failed: %s", dlerror());
        return "";
    }
    typedef Qnn_ErrorHandle_t (*InterfaceGetProvidersFn_t)(
        const QnnInterface_t***, uint32_t*);
    auto getProviders = reinterpret_cast<InterfaceGetProvidersFn_t>(
        dlsym(htpLib, "QnnInterface_getProviders"));
    if (!getProviders) {
        LOGE("probeHtpArch: dlsym QnnInterface_getProviders failed: %s", dlerror());
        dlclose(htpLib);
        return "";
    }
    QnnInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    std::string result;
    if (QNN_SUCCESS == getProviders((const QnnInterface_t***)&providers, &numProviders)) {
        for (uint32_t i = 0; i < numProviders; i++) {
            auto& iface = providers[i];
            if (QNN_API_VERSION_MAJOR != iface->apiVersion.coreApiVersion.major ||
                QNN_API_VERSION_MINOR > iface->apiVersion.coreApiVersion.minor) {
                continue;
            }
            auto& qnn = iface->QNN_INTERFACE_VER_NAME;
            // 单 provider 缺函数指针/取信息失败只跳过该 provider, 不中断循环
            // A provider missing function pointers or failing to report its platform info is skipped,
            // (否则只测第一个 provider, 后续合法 provider 永远不会被探测到)
            // the loop is not aborted (otherwise only the first provider would ever be checked)
            if (!qnn.deviceGetPlatformInfo || !qnn.deviceFreePlatformInfo) continue;
            const QnnDevice_PlatformInfo_t* platformInfo = nullptr;
            if (QNN_SUCCESS != qnn.deviceGetPlatformInfo(nullptr, &platformInfo)) continue;
            const char* name = nullptr;
            const char* archName = nullptr;
            if (platformInfo->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
                for (uint32_t d = 0; d < platformInfo->v1.numHwDevices; d++) {
                    auto& hw = platformInfo->v1.hwDevices[d];
                    if (hw.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) continue;
                    // deviceInfoExtension 由 HTP backend 填充为 QnnHtpDevice_DeviceInfoExtension_t
                    // deviceInfoExtension is filled by the HTP backend as a QnnHtpDevice_DeviceInfoExtension_t
                    auto* ext =
                        (const QnnHtpDevice_DeviceInfoExtension_t*)hw.v1.deviceInfoExtension;
                    if (!ext) continue;
                    if (ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
                        if (ext->onChipDevice.arch == QNN_HTP_DEVICE_ARCH_V68) archName = "V68";
                        else if (ext->onChipDevice.arch == QNN_HTP_DEVICE_ARCH_V69) archName = "V69";
                        else if (ext->onChipDevice.arch == QNN_HTP_DEVICE_ARCH_V73) archName = "V73";
                        else if (ext->onChipDevice.arch == QNN_HTP_DEVICE_ARCH_V75) archName = "V75";
                        else if (ext->onChipDevice.arch == QNN_HTP_DEVICE_ARCH_V79) archName = "V79";
                        else if (ext->onChipDevice.arch == QNN_HTP_DEVICE_ARCH_V81) archName = "V81";
                        else if (ext->onChipDevice.arch == QNN_HTP_DEVICE_ARCH_V85) archName = "V85";
                        else if (ext->onChipDevice.arch == QNN_HTP_DEVICE_ARCH_V89) archName = "V89";
                        if (archName) {
                            LOGI("probeHtpArch: device %u arch=%s socModel=0x%x",
                                 hw.v1.deviceId, archName, ext->onChipDevice.socModel);
                            name = archName;
                        }
                        break;
                    }
                }
            }
            qnn.deviceFreePlatformInfo(nullptr, platformInfo);
            if (name) result = name;
            break;
        }
    }
    dlclose(htpLib);
    if (result.empty()) LOGE("probeHtpArch: no HTP device arch detected");
    return result;
}

// modelType code → runtime 引用 (用于 load/free/compile 路由)
// modelType code -> runtime reference (used for load/free/compile routing)
static std::unique_ptr<qnn::HtpRuntime>& runtimeFor(const std::string& modelType) {
    if (modelType == "pe")      return g_peRuntime;
    if (modelType == "ie")      return g_ieRuntime;
    if (modelType == "rest_a")  return g_restARuntime;
    if (modelType == "rest_b")  return g_restBRuntime;
    if (modelType == "rest_c")  return g_restCRuntime;
    return g_peRuntime; // 默认 / default
}

// 张量名 → 安全文件名 (替换 / 为 _)
// Tensor name -> safe file name (replaces / with _)
static std::string sanitizeTensorName(const std::string& name) {
    std::string r = name;
    for (auto& c : r) if (c == '/' || c == '\\' || c == ':') c = '_';
    return r;
}

// 获取字符串字段
// Converts a jstring field
static std::string jstrToString(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* cstr = env->GetStringUTFChars(jstr, nullptr);
    std::string s(cstr);
    env->ReleaseStringUTFChars(jstr, cstr);
    return s;
}

// 本进程当前 RSS (kB), 用于定位各推理阶段的内存峰值归属 (读 /proc/self/status)
// Current process RSS (kB) to attribute memory peaks to each inference stage (reads /proc/self/status)
static size_t selfVmRSS_kB() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t rss = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

// 读取设备可用内存 (kB), 用于自适应缓存大小 (读 /proc/meminfo)
// Reads the device's available memory (kB) for adaptive cache sizing (reads /proc/meminfo)
static size_t readMemAvailableKB() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t mem = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, "%zu", &mem);
            break;
        }
    }
    fclose(f);
    return mem;
}

// 根据设备可用内存自适应计算缓存阈值和预算
// Computes the adaptive cache threshold and budget based on available device memory
// 策略: 取可用内存的 2% 作为单张量阈值, 4% 作为总预算, 但不低于 2MB/4MB 也不超过 16MB/32MB
// Strategy: 2% of available memory as the per-tensor threshold, 4% as the total budget;
// clamped to [2MB, 16MB] and [4MB, 32MB] respectively
static size_t adaptiveCacheThreshold() {
    size_t availKB = readMemAvailableKB();
    if (availKB == 0) return 8 * 1024 * 1024; // 默认 8MB / default 8MB
    size_t availBytes = availKB * 1024;
    size_t threshold = availBytes / 50; // 2% of available / 2% 可用内存
    if (threshold < 2 * 1024 * 1024) threshold = 2 * 1024 * 1024;   // 最小 2MB / min 2MB
    if (threshold > 16 * 1024 * 1024) threshold = 16 * 1024 * 1024; // 最大 16MB / max 16MB
    return threshold;
}

static size_t adaptiveCacheBudget() {
    size_t availKB = readMemAvailableKB();
    if (availKB == 0) return 16 * 1024 * 1024; // 默认 16MB / default 16MB
    size_t availBytes = availKB * 1024;
    size_t budget = availBytes / 25; // 4% of available / 4% 可用内存
    if (budget < 4 * 1024 * 1024) budget = 4 * 1024 * 1024;      // 最小 4MB / min 4MB
    if (budget > 32 * 1024 * 1024) budget = 32 * 1024 * 1024;    // 最大 32MB / max 32MB
    return budget;
}

// 检查 JNI 异常并清理 (返回 true 表示有异常)
// Checks and clears JNI exceptions (returns true if an exception was pending)
static bool checkJniException(JNIEnv* env) {
    if (!env) return false;
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        return true;
    }
    return false;
}

// ============== JNI 方法实现 ==============
// ============== JNI method implementations ==============

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_jvm = vm;
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    return JNI_VERSION_1_6;
}

// ====== 生命周期 ======
// ====== Lifecycle ======

// 探测设备实际 HTP 架构 ("V79" 等)。自动选择对应 skel 库,
// Probes the device's actual HTP architecture (e.g. "V79") so the matching skel library
// 无需手工指定 HTP 版本。
// is selected automatically without hand-picking the HTP version.
JNIEXPORT jstring JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_probeHtpArch(JNIEnv* env, jobject thiz,
                                                jstring jLibDir) {
    std::string libDir = jstrToString(env, jLibDir);
    std::string arch = probeHtpArch(libDir);
    return env->NewStringUTF(arch.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_nativeInit(JNIEnv* env, jobject thiz,
                                                jstring jLibDir, jstring jSkelDir, jstring jArch) {
    std::lock_guard<std::mutex> lock(g_initMutex);

    // 检查是否已初始化, 避免重复初始化导致资源泄漏
    // Skip if already initialized to avoid leaking resources on double init
    if (g_qnnInitialized) {
        LOGI("nativeInit: already initialized, skip duplicate init");
        return JNI_TRUE;
    }

    std::string libDir = jstrToString(env, jLibDir);
    std::string skelDir = jstrToString(env, jSkelDir);
    std::string archStr = jstrToString(env, jArch);
    qnn::HtpArch arch = parseArch(archStr);

    LOGI("nativeInit: libDir=%s skelDir=%s arch=%s", libDir.c_str(), skelDir.c_str(), archStr.c_str());

    // 创建五个运行时 (PE/IE/REST_A/B/C 各自独立 context, 共享同一 backend/device)
    // Create the five runtimes (PE/IE/REST_A/B/C each own a context and share one backend/device)
    auto initOne = [&](std::unique_ptr<qnn::HtpRuntime>& rt, const char* tag) -> int {
        rt = std::make_unique<qnn::HtpRuntime>(&g_qnnShared);
        int r = rt->init(libDir, skelDir, arch, g_perfCfg);
        if (r != 0)         LOGE("%s runtime init failed: %d", tag, r);
        return r;
    };
    if (initOne(g_peRuntime,     "PE")     != 0) {
        g_peRuntime.reset();
        return JNI_FALSE;
    }
    if (initOne(g_ieRuntime,     "IE")     != 0) {
        g_peRuntime.reset();
        g_ieRuntime.reset();
        return JNI_FALSE;
    }
    if (initOne(g_restARuntime,  "REST_A") != 0) {
        g_peRuntime.reset();
        g_ieRuntime.reset();
        g_restARuntime.reset();
        return JNI_FALSE;
    }
    if (initOne(g_restBRuntime,  "REST_B") != 0) {
        g_peRuntime.reset();
        g_ieRuntime.reset();
        g_restARuntime.reset();
        g_restBRuntime.reset();
        return JNI_FALSE;
    }
    if (initOne(g_restCRuntime,  "REST_C") != 0) {
        g_peRuntime.reset();
        g_ieRuntime.reset();
        g_restARuntime.reset();
        g_restBRuntime.reset();
        g_restCRuntime.reset();
        return JNI_FALSE;
    }

    g_dlcCompiler = std::make_unique<qnn::DlcCompiler>();

    // 根据设备可用内存自适应设置 REST 缓存参数
    // Set REST cache parameters adaptively based on available device memory
    g_restCacheMaxBytes = adaptiveCacheThreshold();
    g_restCacheTotalBudget = adaptiveCacheBudget();
    LOGI("REST cache: threshold=%zuMB budget=%zuMB (available=%zuMB)",
         g_restCacheMaxBytes / (1024*1024), g_restCacheTotalBudget / (1024*1024),
         readMemAvailableKB() / 1024);

    // 注册 SHARP core 进度回调
    // Register the SHARP core progress callback
    sharp_set_progress_callback(sharpProgressCallback);

    g_qnnInitialized = true;
    LOGI("QNN runtime init OK (5 runtimes)");
    return JNI_TRUE;
}

// 设置 HTP 性能配置 (须在 nativeInit 前调用; 已初始化时仅提示)
// Sets the HTP performance config (call before nativeInit; only logs when already initialized)
// type: 0=锁角模式 (lockedCorner), 1=自动调角模式 (min/target/max + dcvsMode)
// type: 0 = locked-corner mode (lockedCorner), 1 = adaptive mode (min/target/max + dcvsMode)
// 电压角/DCVS 模式取值与 QnnHtpPerfInfrastructure 枚举一致 (角 0x20~0xA0, 模式 0x1~0x20)
// Corner/DCVS values match the QnnHtpPerfInfrastructure enums (corners 0x20~0xA0, modes 0x1~0x20)
JNIEXPORT void JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_setPerfConfig(JNIEnv* env, jobject thiz,
                                                 jint jType, jint jLocked, jint jMin,
                                                 jint jTarget, jint jMax, jint jDcvsMode) {
    std::lock_guard<std::mutex> lock(g_initMutex);
    if (jType != 0 && jType != 1) {
        LOGE("setPerfConfig: invalid mode type %d (only 0=locked/1=auto-range)", jType);
        return;
    }
    g_perfCfg.type = jType;
    g_perfCfg.lockedCorner = static_cast<uint32_t>(jLocked);
    g_perfCfg.minCorner = static_cast<uint32_t>(jMin);
    g_perfCfg.targetCorner = static_cast<uint32_t>(jTarget);
    g_perfCfg.maxCorner = static_cast<uint32_t>(jMax);
    g_perfCfg.dcvsMode = static_cast<uint32_t>(jDcvsMode);
    // 非法值由 native 层 (createSharedState) 防御性收敛, 此处仅日志
    // Invalid values are clamped in the native layer (createSharedState); here we only log
    LOGI("setPerfConfig: type=%d (%s) locked=%u min=%u target=%u max=%u dcvsMode=%u",
         jType, jType == 0 ? "locked" : "auto-range", jLocked, jMin, jTarget, jMax, jDcvsMode);
    if (g_qnnInitialized) {
        LOGI("setPerfConfig: already initialized; changes apply after restart");
    } else {
        LOGI("setPerfConfig: will take effect at nativeInit");
    }
}

JNIEXPORT void JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_nativeDestroy(JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_initMutex);

    // 检查是否已初始化, 避免重复销毁
    // Skip when not initialized to avoid double teardown
    if (!g_qnnInitialized) {
        LOGI("nativeDestroy: not initialized, skip");
        return;
    }

    // 先释放 DlcCompiler (其析构可能调用 systemDlcFree, 依赖 libQnnSystem.so 仍被加载)
    // Free the DlcCompiler first (its destructor may call systemDlcFree while libQnnSystem.so is still loaded)
    g_dlcCompiler.reset();
    // 再释放 runtimes: 释放各自 context+graph; 最后一个 runtime 释放共享 backend/device/log
    // Then free the runtimes: each frees its own context+graph; the last one frees the shared backend/device/log
    if (g_peRuntime)     { g_peRuntime->freeContext();     g_peRuntime.reset(); }
    if (g_ieRuntime)     { g_ieRuntime->freeContext();     g_ieRuntime.reset(); }
    if (g_restARuntime)  { g_restARuntime->freeContext();  g_restARuntime.reset(); }
    if (g_restBRuntime)  { g_restBRuntime->freeContext();  g_restBRuntime.reset(); }
    if (g_restCRuntime)  { g_restCRuntime->freeContext();  g_restCRuntime.reset(); }
    if (g_callbackObj) {
        env->DeleteGlobalRef(g_callbackObj);
        g_callbackObj = nullptr;
    }
    g_qnnInitialized = false;
    LOGI("QNN runtime destroyed");
}

// ====== 模型管理 ======
// ====== Model management ======

JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_loadContextBinary(JNIEnv* env, jobject thiz,
                                                      jstring jModelType, jstring jBinPath) {
    std::string modelType = jstrToString(env, jModelType);
    std::string binPath = jstrToString(env, jBinPath);

    LOGI("loadContextBinary: type=%s path=%s", modelType.c_str(), binPath.c_str());

    auto& runtime = runtimeFor(modelType);
    if (!runtime) {         LOGE("runtime not ready"); return JNI_FALSE; }

    std::string graphName;
    int ret = runtime->loadFromBinary(binPath, graphName);
    if (ret != 0) {
        LOGE("loadFromBinary failed: %d", ret);
        return JNI_FALSE;
    }
    LOGI("model loaded, graph=%s RSS=%zuKB", graphName.c_str(), selfVmRSS_kB());
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_compileDlc(JNIEnv* env, jobject thiz,
                                               jstring jModelType, jstring jDlcPath, jstring jOutBinPath) {
    // 与 nativeDestroy/nativeInit 互斥, 防止编译期间销毁资源 (悬垂访问)
    // Mutually exclusive with nativeDestroy/nativeInit so resources cannot be torn down during compile
    std::lock_guard<std::mutex> lock(g_initMutex);

    if (!g_qnnInitialized) {
        LOGE("compileDlc: QNN not initialized");
        return JNI_FALSE;
    }

    std::string modelType = jstrToString(env, jModelType);
    std::string dlcPath = jstrToString(env, jDlcPath);
    std::string outBinPath = jstrToString(env, jOutBinPath);

    LOGI("compileDlc: type=%s dlc=%s out=%s", modelType.c_str(), dlcPath.c_str(), outBinPath.c_str());

    auto& runtime = runtimeFor(modelType);
    if (!runtime || !g_dlcCompiler) {         LOGE("runtime/compiler not ready"); return JNI_FALSE; }

    int ret = g_dlcCompiler->compile(*runtime, dlcPath, outBinPath, nullptr);
    if (ret != 0) {
        LOGE("DLC compile failed: %d", ret);
        return JNI_FALSE;
    }
    LOGI("DLC compile OK: %s", outBinPath.c_str());
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_cancelCompile(JNIEnv* env, jobject thiz) {
    if (!g_dlcCompiler) return;
    g_dlcCompiler->requestCancel();
    LOGI("cancelCompile: DLC compile cancel requested");
}

JNIEXPORT void JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_freeContext(JNIEnv* env, jobject thiz, jstring jModelType) {
    std::string modelType = jstrToString(env, jModelType);
    auto& runtime = runtimeFor(modelType);
    if (runtime) runtime->freeContext();
}

// ====== 推理 Pipeline ======
// ====== Inference pipeline ======

JNIEXPORT jfloatArray JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_prepImage(JNIEnv* env, jobject thiz,
                                              jstring jImagePath, jstring jOutRawPath) {
    std::string imagePath = jstrToString(env, jImagePath);
    std::string outRawPath = jstrToString(env, jOutRawPath);

    LOGI("prepImage entry RSS=%zuKB", selfVmRSS_kB());
    float fpx = 0, dfactor = 0;
    int origW = 0, origH = 0;
    int ret = sharp_prep_image(imagePath.c_str(), outRawPath.c_str(), &fpx, &dfactor, &origW, &origH);
    LOGI("prepImage exit RSS=%zuKB", selfVmRSS_kB());
    if (ret != 0) {
        LOGE("prepImage failed: %d", ret);
        cbError(0, "err_prep_image_failed");
        return nullptr;
    }

    // 返回 [fpx, dfactor, origW, origH]
    // Returns [fpx, dfactor, origW, origH]
    jfloatArray result = env->NewFloatArray(4);
    if (!result || checkJniException(env)) return nullptr;
    float vals[] = {fpx, dfactor, (float)origW, (float)origH};
    env->SetFloatArrayRegion(result, 0, 4, vals);
    return result;
}

JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_runPre(JNIEnv* env, jobject thiz,
                                           jstring jImageRawPath, jstring jWorkDir) {
    std::string imageRawPath = jstrToString(env, jImageRawPath);
    std::string workDir = jstrToString(env, jWorkDir);

    LOGI("runPre entry RSS=%zuKB", selfVmRSS_kB());
    int ret = sharp_pre(imageRawPath.c_str(), workDir.c_str());
    LOGI("runPre exit RSS=%zuKB", selfVmRSS_kB());
    if (ret != 0) {
        LOGE("pre failed: %d", ret);
        cbError(1, "err_pre_failed");
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

// PE 推理: 35 个 patch
// PE inference: 35 patches
JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_runPatchEncoder(JNIEnv* env, jobject thiz, jstring jWorkDir) {
    std::string workDir = jstrToString(env, jWorkDir);

    if (!g_peRuntime || !g_peRuntime->isReady()) {
        LOGE("PE runtime not ready");
        cbError(2, "err_pe_not_loaded");
        return JNI_FALSE;
    }

    LOGI("runPatchEncoder entry RSS=%zuKB", selfVmRSS_kB());
    auto t0 = std::chrono::steady_clock::now();

    // 输出目录
    // Output directory
    std::string outDir = workDir + "/out_pe";
    ensureDir(outDir);

    const int NUM_PATCHES = 35;
    const auto& inputInfos = g_peRuntime->getInputInfos();
    const auto& outputInfos = g_peRuntime->getOutputInfos();

    if (inputInfos.empty() || outputInfos.empty()) {
        LOGE("PE tensor info empty");
        cbError(2, "err_pe_tensor_info_empty");
        return JNI_FALSE;
    }

    // 输入: [1,3,384,384] NCHW, float32 (DLC 输入布局, 由 inputInfos[0].dims 给出)
    // Input: [1,3,384,384] NCHW float32 (the DLC input layout, given by inputInfos[0].dims)
    // 输出: patch_features [1,1024,24,24], latent0 [1,1024,24,24], latent1 [1,1024,24,24]
    // Outputs: patch_features [1,1024,24,24], latent0 [1,1024,24,24], latent1 [1,1024,24,24]

    // 输出缓冲: 循环外一次性分配, 35 个 patch 复用 (避免反复分配/释放)
    // Output buffers: allocated once outside the loop and reused across the 35 patches (no repeated allocate/free)
    std::vector<qnn::Tensor> outputs(outputInfos.size());
    std::vector<std::vector<float>> outBuffers(outputInfos.size());
    for (size_t j = 0; j < outputInfos.size(); j++) {
        outBuffers[j].resize(qnn::calculateElementCount(outputInfos[j].dims));
        outputs[j].name = outputInfos[j].name;
        outputs[j].dims = outputInfos[j].dims;
        outputs[j].data = outBuffers[j].data();
        outputs[j].count = outBuffers[j].size();
    }

    for (int i = 0; i < NUM_PATCHES; i++) {
        // 读取 patch raw
        // Read the patch raw file
        char patchName[64];
        snprintf(patchName, sizeof(patchName), "%s/patch_p%04d.raw", workDir.c_str(), i);
        auto patchData = readRawFile(patchName);
        if (patchData.empty()) {
            LOGE("read patch failed: %s", patchName);
            cbError(2, "err_read_patch_failed");
            return JNI_FALSE;
        }

        // 构造输入张量
        // Build the input tensor
        qnn::Tensor input;
        input.name = inputInfos[0].name;
        input.dims = inputInfos[0].dims;
        input.data = patchData.data();
        input.count = patchData.size();

        // 执行推理 (输出缓冲复用, execute 每次全量覆盖)
        // Run inference (the output buffers are reused; execute overwrites them fully each time)
        int ret = g_peRuntime->execute({input}, outputs);
        if (ret != 0) {
            LOGE("PE inference failed patch %d: %d", i, ret);
            cbError(2, "err_pe_infer_failed");
            return JNI_FALSE;
        }

        // 写输出到 out_pe/Result_i/
        // Write outputs to out_pe/Result_i/
        char resultDir[128];
        snprintf(resultDir, sizeof(resultDir), "%s/out_pe/Result_%d", workDir.c_str(), i);
        ensureDir(resultDir);
        for (size_t j = 0; j < outputs.size(); j++) {
            char outPath[256];
            snprintf(outPath, sizeof(outPath), "%s/%s.raw", resultDir, outputs[j].name.c_str());
            if (!writeRawFile(outPath, outputs[j].data, outputs[j].count)) {
                LOGE("PE patch %d write failed: %s", i, outPath);
                cbError(2, "err_pe_write_failed");
                return JNI_FALSE;
            }
        }

        // 进度回调
        // Progress callback
        auto t1 = std::chrono::steady_clock::now();
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        char detail[64];
        snprintf(detail, sizeof(detail), "patch %d/%d", i + 1, NUM_PATCHES);
        cbProgress(2, i + 1, NUM_PATCHES, ms, detail);
    }

    LOGI("runPatchEncoder exit RSS=%zuKB", selfVmRSS_kB());
    return JNI_TRUE;
}

// IE 推理: 1 次
// IE inference: a single run
JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_runImageEncoder(JNIEnv* env, jobject thiz, jstring jWorkDir) {
    std::string workDir = jstrToString(env, jWorkDir);

    if (!g_ieRuntime || !g_ieRuntime->isReady()) {
        LOGE("IE runtime not ready");
        cbError(3, "err_ie_not_loaded");
        return JNI_FALSE;
    }

    // 读取 x2.raw
    // Read x2.raw
    std::string x2Path = workDir + "/x2.raw";
    auto x2Data = readRawFile(x2Path);
    if (x2Data.empty()) {
        LOGE("read x2.raw failed");
        cbError(3, "err_read_x2_failed");
        return JNI_FALSE;
    }

    LOGI("runImageEncoder entry RSS=%zuKB", selfVmRSS_kB());

    const auto& inputInfos = g_ieRuntime->getInputInfos();
    const auto& outputInfos = g_ieRuntime->getOutputInfos();

    // 构造输入
    // Build the input
    qnn::Tensor input;
    input.name = inputInfos[0].name;
    input.dims = inputInfos[0].dims;
    input.data = x2Data.data();
    input.count = x2Data.size();

    // 构造输出
    // Build the outputs
    std::vector<qnn::Tensor> outputs(outputInfos.size());
    std::vector<std::vector<float>> outBuffers(outputInfos.size());
    for (size_t j = 0; j < outputInfos.size(); j++) {
        outBuffers[j].resize(qnn::calculateElementCount(outputInfos[j].dims));
        outputs[j].name = outputInfos[j].name;
        outputs[j].dims = outputInfos[j].dims;
        outputs[j].data = outBuffers[j].data();
        outputs[j].count = outBuffers[j].size();
    }

    // 执行推理
    // Run inference
    int ret = g_ieRuntime->execute({input}, outputs);
    if (ret != 0) {
        LOGE("IE inference failed: %d", ret);
        cbError(3, "err_ie_infer_failed");
        return JNI_FALSE;
    }

    // 写输出到 out_ie/Result_0/
    // Write outputs to out_ie/Result_0/
    std::string resultDir = workDir + "/out_ie/Result_0";
    ensureDir(resultDir);
    for (size_t j = 0; j < outputs.size(); j++) {
        std::string outPath = resultDir + "/" + outputs[j].name + ".raw";
        if (!writeRawFile(outPath, outputs[j].data, outputs[j].count)) {
            LOGE("IE write failed: %s", outPath.c_str());
            cbError(3, "err_ie_write_failed");
            return JNI_FALSE;
        }
    }

    LOGI("runImageEncoder exit RSS=%zuKB", selfVmRSS_kB());
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_runMerge(JNIEnv* env, jobject thiz,
                                             jstring jWorkDir, jstring jPeOutDir, jstring jIeOutDir) {
    std::string workDir = jstrToString(env, jWorkDir);
    std::string peOutDir = jstrToString(env, jPeOutDir);
    std::string ieOutDir = jstrToString(env, jIeOutDir);

    LOGI("runMerge entry RSS=%zuKB", selfVmRSS_kB());
    int ret = sharp_merge(workDir.c_str(), peOutDir.c_str(), ieOutDir.c_str());
    LOGI("runMerge exit RSS=%zuKB", selfVmRSS_kB());
    if (ret != 0) {
        LOGE("merge failed: %d", ret);
        cbError(4, "err_merge_failed");
        return JNI_FALSE;
    }

    return JNI_TRUE;
}

// ====== REST 三段推理 ======
// ====== REST three-segment inference ======

// 通用: 按输入张量名查找数据 (缓存→目录搜索→显式路径), 执行推理, 输出写文件+缓存
// Generic: resolves each input tensor by name (cache -> directory search -> explicit path),
// runs inference, writes each output to a file and optionally the cache
// explicitInputs: 张量名 → .raw 文件路径 (特殊输入, 如 image/disparity)
// explicitInputs: tensor name -> .raw file path (special inputs, e.g. image/disparity)
// searchDirs: 输入文件搜索目录 (按 sanitized 张量名匹配 <name>.raw)
// searchDirs: directories to search for input files (matching <name>.raw via the sanitized tensor name)
// outputDir: 输出文件目录
// outputDir: output file directory
// 输出文件名 = sanitizeTensorName(张量名) + ".raw"
// Output file name = sanitizeTensorName(tensor name) + ".raw"
static bool runRestInference(qnn::HtpRuntime* rt,
                             const std::map<std::string, std::string>& explicitInputs,
                             const std::vector<std::string>& searchDirs,
                             const std::string& outputDir,
                             int stageId, const char* stageName) {
    if (!rt || !rt->isReady()) {
        LOGE("%s runtime not ready", stageName);
        std::string msgNotReady = std::string("err_runtime_not_ready|") + stageName;
        cbError(stageId, msgNotReady.c_str());
        return false;
    }
    ensureDir(outputDir);

    const auto& inputInfos = rt->getInputInfos();
    const auto& outputInfos = rt->getOutputInfos();
    if (inputInfos.empty() || outputInfos.empty()) {
        LOGE("%s tensor info empty", stageName);
        std::string msgTensorEmpty = std::string("err_tensor_info_empty|") + stageName;
        cbError(stageId, msgTensorEmpty.c_str());
        return false;
    }

    // 构造输入: 依次尝试 显式路径 → 内存缓存 → 目录搜索
    // Resolve inputs: explicit path -> memory cache -> directory search
    // 量化输入 (UFIXED_16/8) 从 float 文件流式读取并直接量化, float 驻留仅 4MB
    // Quantized inputs (UFIXED_16/8) are streamed from float files and quantized on the fly (only 4MB of float resident)
    std::vector<qnn::Tensor> inputs(inputInfos.size());
    std::vector<std::vector<float>> inputBuffers(inputInfos.size());
    std::vector<std::vector<uint8_t>> quantInputBuffers(inputInfos.size());
    const size_t IN_CHUNK = 1u << 20;  // 1M 元素/块 (4MB float 临时缓冲) / 1M elements per chunk (4MB float scratch)
    for (size_t i = 0; i < inputInfos.size(); i++) {
        const auto& info = inputInfos[i];
        bool found = false;
        const bool isQuant = (info.dataType == QNN_DATATYPE_UFIXED_POINT_16 ||
                              info.dataType == QNN_DATATYPE_UFIXED_POINT_8);
        const size_t elemCount = qnn::calculateElementCount(info.dims);

        // 流式读取 float 文件并直接量化为 uint16/uint8
        // Streams a float file and quantizes straight into uint16/uint8
        auto loadQuantFromFile = [&](const std::string& path) -> bool {
            std::ifstream f(path, std::ios::binary);
            if (!f) return false;
            size_t fileBytes = 0;
            f.seekg(0, std::ios::end); fileBytes = (size_t)f.tellg(); f.seekg(0, std::ios::beg);
            if (fileBytes < elemCount * sizeof(float)) return false;
            size_t outBytes = (info.dataType == QNN_DATATYPE_UFIXED_POINT_16)
                                  ? elemCount * sizeof(uint16_t) : elemCount * sizeof(uint8_t);
            quantInputBuffers[i].resize(outBytes);
            std::vector<float> tmp(std::min(IN_CHUNK, elemCount));
            size_t done = 0;
            while (done < elemCount) {
                size_t n = std::min(IN_CHUNK, elemCount - done);
                f.read(reinterpret_cast<char*>(tmp.data()), n * sizeof(float));
                if (!f && f.gcount() != (std::streamsize)(n * sizeof(float))) return false;
                if (info.dataType == QNN_DATATYPE_UFIXED_POINT_16) {
                    qnn::floatToUfixed16(tmp.data(),
                        reinterpret_cast<uint16_t*>(quantInputBuffers[i].data()) + done,
                        n, info.scale, info.offset);
                } else {
                    qnn::floatToUfixed8(tmp.data(),
                        quantInputBuffers[i].data() + done,
                        n, info.scale, info.offset);
                }
                done += n;
            }
            return true;
        };

        // 1. 显式路径 (精确名称匹配)
        // 1. Explicit path (exact name match)
        auto it = explicitInputs.find(info.name);
        if (it != explicitInputs.end()) {
            if (isQuant) {
                if (!loadQuantFromFile(it->second)) {
                    LOGE("%s quantized read failed: %s", stageName, it->second.c_str());
                    std::string msgReadFail = std::string("err_tensor_read_failed|") + stageName + "|" + it->second;
                    cbError(stageId, msgReadFail.c_str());
                    return false;
                }
            } else {
                inputBuffers[i] = readRawFile(it->second);
                if (inputBuffers[i].empty()) {
                    LOGE("%s read failed: %s", stageName, it->second.c_str());
                    std::string msgReadFail = std::string("err_tensor_read_failed|") + stageName + "|" + it->second;
                    cbError(stageId, msgReadFail.c_str());
                    return false;
                }
            }
            found = true;
        }

        // 2. 内存缓存 (精确名称匹配, 命中后移除)
        // 2. Memory cache (exact name match; removed after a hit)
        if (!found) {
            auto cit = g_restCache.find(info.name);
            if (cit != g_restCache.end() && cit->second.size() >= elemCount) {
                g_restCacheBytes -= cit->second.size() * sizeof(float);
                if (isQuant) {
                    // 缓存内是 float 数据, 量化输入需先转为 uint16/uint8,
                    // Cache entries are float; quantized inputs must first be converted to uint16/uint8,
                    // 否则 quantInputBuffers[i] 为空 → execute 阶段将按空指针拷贝
                    // otherwise quantInputBuffers[i] would stay empty and execute would copy from a null pointer
                    size_t outBytes = (info.dataType == QNN_DATATYPE_UFIXED_POINT_16)
                                          ? elemCount * sizeof(uint16_t)
                                          : elemCount * sizeof(uint8_t);
                    quantInputBuffers[i].resize(outBytes);
                    if (info.dataType == QNN_DATATYPE_UFIXED_POINT_16) {
                        qnn::floatToUfixed16(cit->second.data(),
                            reinterpret_cast<uint16_t*>(quantInputBuffers[i].data()),
                            elemCount, info.scale, info.offset);
                    } else {
                        qnn::floatToUfixed8(cit->second.data(),
                            quantInputBuffers[i].data(),
                            elemCount, info.scale, info.offset);
                    }
                }
                inputBuffers[i] = std::move(cit->second);
                g_restCache.erase(cit);
                found = true;
                LOGD("%s cache hit: %s", stageName, info.name.c_str());
            }
        }

        // 3. 目录搜索 (按 sanitized 名称查找 <name>.raw)
        // 3. Directory search (looks for <name>.raw via the sanitized name)
        if (!found) {
            std::string fname = sanitizeTensorName(info.name) + ".raw";
            for (const auto& dir : searchDirs) {
                std::string candidate = dir + "/" + fname;
                std::ifstream test(candidate);
                if (test.good()) {
                    if (isQuant) {
                        if (!loadQuantFromFile(candidate)) {
                            LOGE("%s quantized read failed: %s", stageName, candidate.c_str());
                            std::string msgReadFail = std::string("err_tensor_read_failed|") + stageName + "|" + candidate;
                            cbError(stageId, msgReadFail.c_str());
                            return false;
                        }
                    } else {
                        inputBuffers[i] = readRawFile(candidate);
                        if (inputBuffers[i].empty()) {
                            LOGE("%s read failed: %s", stageName, candidate.c_str());
                            std::string msgReadFail = std::string("err_tensor_read_failed|") + stageName + "|" + candidate;
                            cbError(stageId, msgReadFail.c_str());
                            return false;
                        }
                    }
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            LOGE("%s input tensor '%s' file not found", stageName, info.name.c_str());
            std::string msgInputMissing = std::string("err_tensor_input_missing|") + stageName + "|" + info.name;
            cbError(stageId, msgInputMissing.c_str());
            return false;
        }

        inputs[i].name = info.name;
        inputs[i].dims = info.dims;
        inputs[i].quantized = isQuant;
        if (isQuant) {
            inputs[i].data = reinterpret_cast<float*>(quantInputBuffers[i].data());
            inputs[i].count = elemCount;
        } else {
            inputs[i].data = inputBuffers[i].data();
            inputs[i].count = inputBuffers[i].size();
        }
    }

    // 构造输出: keepOutputQuantized 模式下不预分配 float 缓冲,
    // Build the outputs: in keepOutputQuantized mode no float buffers are pre-allocated;
    // 量化原始数据 (uint16/uint8) 留在 runtime 内部, 执行后逐张反量化落盘
    // raw quantized data (uint16/uint8) stays inside the runtime and is dequantized per tensor when persisted
    std::vector<qnn::Tensor> outputs(outputInfos.size());
    for (size_t j = 0; j < outputInfos.size(); j++) {
        outputs[j].name = outputInfos[j].name;
        outputs[j].dims = outputInfos[j].dims;
        outputs[j].data = nullptr;
        outputs[j].count = qnn::calculateElementCount(outputInfos[j].dims);
    }

    int ret = rt->execute(inputs, outputs, /*keepOutputQuantized=*/true);
    if (ret != 0) {
        LOGE("%s inference failed: %d", stageName, ret);
        std::string msgInferFail = std::string("err_infer_failed|") + stageName;
        cbError(stageId, msgInferFail.c_str());
        return false;
    }

    // QnnGraph_execute 为同步阻塞调用, 返回时输入已消费完毕:
    // QnnGraph_execute is synchronous and blocking; by the time it returns the inputs are fully consumed:
    // 立即归还输入缓冲 (省下写文件阶段内存), 降低本段峰值
    // free the input buffers right away (saving the write-phase memory) to lower this segment's peak
    inputBuffers.clear();
    quantInputBuffers.clear();
    inputs.clear();

    // 写输出: 文件 (始终) + 内存缓存 (受预算约束, 供下一段使用)
    // Write outputs: to file (always) + to the memory cache (budget-bound, for the next segment)
    // 逐张: 量化数据 → 分块反量化流式写文件, 完成后立即释放该张量化缓冲,
    // Per tensor: dequantize in chunks and stream to file, then release that tensor's quantized buffer,
    // 避免全部 float 输出同时驻留 (rest_a 输出合计 ~1GB, 峰值可降一半)
    // avoiding all float outputs resident at once (rest_a outputs total ~1GB; the peak drops by half)
    const size_t CHUNK = 1u << 20;  // 1M 元素/块 (4MB float 临时缓冲) / 1M elements per chunk (4MB float scratch)
    // 中途失败返回时释放尚未落盘的量化缓冲, 避免数百 MB 滞留至下次 execute
    // On early failure, release the quantized buffers not yet persisted to avoid hundreds of MB lingering into the next execute
    auto releaseUnwritten = [&](size_t from) {
        for (size_t k = from; k < outputs.size(); k++) rt->releaseQuantizedOutput(k);
    };
    for (size_t j = 0; j < outputs.size(); j++) {
        const uint8_t* raw = rt->getQuantizedOutputData(j);
        if (!raw) {
            LOGE("%s output tensor '%s' data missing", stageName, outputs[j].name.c_str());
            std::string msgOutMissing = std::string("err_output_missing|") + stageName;
            cbError(stageId, msgOutMissing.c_str());
            releaseUnwritten(j);
            return false;
        }

        // 定位该输出的量化参数 (scale/offset)
        // Locate this output's quantization params (scale/offset)
        const qnn::HtpRuntime::TensorInfo* oinfo = nullptr;
        for (const auto& oi : outputInfos) {
            if (oi.name == outputs[j].name) { oinfo = &oi; break; }
        }
        if (!oinfo) {
            LOGE("%s output tensor '%s' metadata missing", stageName, outputs[j].name.c_str());
            std::string msgMetaMissing = std::string("err_output_meta_missing|") + stageName;
            cbError(stageId, msgMetaMissing.c_str());
            releaseUnwritten(j);
            return false;
        }

        // 分块反量化 → 流式写文件 (float 临时缓冲仅 CHUNK 大小, 非全量 1GB)
        // Chunked dequantization -> streamed file write (only CHUNK-sized float scratch, not the full 1GB)
        {
            std::string fname = sanitizeTensorName(outputs[j].name) + ".raw";
            std::string outPath = outputDir + "/" + fname;
            std::ofstream f(outPath, std::ios::binary);
            if (!f) {
                LOGE("%s cannot write: %s", stageName, outPath.c_str());
                std::string msgWriteFail = std::string("err_write_failed|") + stageName;
                cbError(stageId, msgWriteFail.c_str());
                releaseUnwritten(j);
                return false;
            }
            std::vector<float> tmp(std::min(CHUNK, outputs[j].count));
            size_t done = 0;
            while (done < outputs[j].count) {
                size_t n = std::min(CHUNK, outputs[j].count - done);
                if (oinfo->dataType == QNN_DATATYPE_UFIXED_POINT_16) {
                    qnn::ufixed16ToFloat(
                        reinterpret_cast<const uint16_t*>(raw) + done,
                        tmp.data(), n, oinfo->scale, oinfo->offset);
                } else if (oinfo->dataType == QNN_DATATYPE_UFIXED_POINT_8) {
                    qnn::ufixed8ToFloat(raw + done, tmp.data(), n,
                                        oinfo->scale, oinfo->offset);
                } else {
                    // float32 直拷
                    // float32 copied directly
                    std::memcpy(tmp.data(), reinterpret_cast<const float*>(raw) + done,
                                n * sizeof(float));
                }
                f.write(reinterpret_cast<const char*>(tmp.data()), n * sizeof(float));
                done += n;
            }
        }

        // 小张量缓存到内存 (受总量预算约束), 避免下一段重新读文件
        // Small tensors also go to the memory cache (budget-bound) so the next segment skips a file read
        restCachePut(outputs[j].name, raw, outputs[j].count, oinfo);

        // 该张量已落盘, 归还其量化缓冲 (写文件阶段内存随写出递减)
        // This tensor is persisted, so return its quantized buffer (write-phase memory shrinks as files complete)
        rt->releaseQuantizedOutput(j);
    }
    return true;
}

// REST Seg A: 特征融合
// REST Seg A: feature fusion
// 输入: x_latent0, x_latent1, x0_feat, x1_feat, x2_feat, x_lowres_feat (来自 workDir/)
// Inputs: x_latent0, x_latent1, x0_feat, x1_feat, x2_feat, x_lowres_feat (from workDir/)
// 输出: 边张量 (存到 workDir/rest_a_out/ + 内存缓存)
// Outputs: edge tensors (stored to workDir/rest_a_out/ + the memory cache)
JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_runRestSegA(JNIEnv* env, jobject thiz, jstring jWorkDir) {
    std::string workDir = jstrToString(env, jWorkDir);

    LOGI("runRestSegA entry RSS=%zuKB", selfVmRSS_kB());
    std::map<std::string, std::string> explicitInputs = {
        {"x_latent0",     workDir + "/x_latent0.raw"},
        {"x_latent1",     workDir + "/x_latent1.raw"},
        {"x0_feat",       workDir + "/x0_feat.raw"},
        {"x1_feat",       workDir + "/x1_feat.raw"},
        {"x2_feat",       workDir + "/x2_feat.raw"},
        {"x_lowres_feat", workDir + "/x_lowres_feat.raw"},
    };
    std::string outDir = workDir + "/rest_a_out";

    if (!runRestInference(g_restARuntime.get(), explicitInputs,                           {workDir}, outDir, 5, "Feature Fusion")) {
        return JNI_FALSE;
    }

    LOGI("runRestSegA exit RSS=%zuKB", selfVmRSS_kB());
    return JNI_TRUE;
}

// REST Seg B: 视差估计
// REST Seg B: disparity estimation
// 输入: Seg A 的边张量 (自动从缓存或 rest_a_out/ 匹配, 无需硬编码张量名)
// Inputs: Seg A's edge tensors (matched automatically from the cache or rest_a_out/, no hardcoded tensor names)
// 输出: disparity + 边张量 (存到 workDir/rest_b_out/ + 内存缓存)
// Outputs: disparity + edge tensors (stored to workDir/rest_b_out/ + the memory cache)
JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_runRestSegB(JNIEnv* env, jobject thiz, jstring jWorkDir) {
    std::string workDir = jstrToString(env, jWorkDir);

    // 所有输入来自 Seg A 输出, 通过目录搜索 + 内存缓存自动匹配
    // All inputs come from Seg A's outputs, matched via directory search + the memory cache
    std::map<std::string, std::string> explicitInputs;
    std::string outDir = workDir + "/rest_b_out";

    LOGI("runRestSegB entry RSS=%zuKB", selfVmRSS_kB());
    if (!runRestInference(g_restBRuntime.get(), explicitInputs,
                          {workDir + "/rest_a_out"}, outDir, 6, "Disparity Estimation")) {
        return JNI_FALSE;
    }

    // 查找 disparity 输出并复制到 workDir/disparity.raw (供 Seg C 和 Post 使用)
    // Find the disparity output and copy it to workDir/disparity.raw (used by Seg C and Post)
    const auto& bOutputs = g_restBRuntime->getOutputInfos();
    bool dispFound = false;
    for (const auto& out : bOutputs) {
        if (out.name == "disparity" || out.name.find("disparity") != std::string::npos) {
            std::string src = outDir + "/" + sanitizeTensorName(out.name) + ".raw";
            std::string dst = workDir + "/disparity.raw";
            {
                std::ifstream srcF(src, std::ios::binary);
                std::ofstream dstF(dst, std::ios::binary);
                if (!srcF || !dstF) {
                    LOGE("disparity copy failed");
                    cbError(6, "err_disparity_copy_failed");
                    return JNI_FALSE;
                }
                dstF << srcF.rdbuf();
            }
            // disparity 18.9MB > 缓存阈值 8MB, 永不会进缓存 (restCachePut 直接 return):
            // disparity (18.9MB) exceeds the 8MB cache threshold, so it never enters the cache (restCachePut returns early);
            // Seg C 通过 workDir/disparity.raw 文件读取, 不再重复读 rest_b_out 下的副本
            // Seg C reads workDir/disparity.raw instead of re-reading the copy under rest_b_out
            dispFound = true;
            break;
        }
    }
    if (!dispFound) {
        LOGE("disparity output tensor not found");
        cbError(6, "err_disparity_output_missing");
        return JNI_FALSE;
    }

    LOGI("runRestSegB exit RSS=%zuKB", selfVmRSS_kB());
    return JNI_TRUE;
}

// REST Seg C: 高斯增量
// REST Seg C: gaussian delta
// 输入: image, disparity_factor (显式), Seg A/B 边张量 (自动匹配)
// Inputs: image, disparity_factor (explicit), Seg A/B edge tensors (auto-matched)
// 输出: delta (存到 workDir/rest_c_out/ + 复制到 workDir/delta.raw)
// Outputs: delta (stored to workDir/rest_c_out/ and copied to workDir/delta.raw)
JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_runRestSegC(JNIEnv* env, jobject thiz,
                                                jstring jWorkDir, jfloat jFpx, jint jOrigW) {
    std::string workDir = jstrToString(env, jWorkDir);

    // disparity_factor = fpx / origW (与 sharp_post 的 d_factor 一致)
    // disparity_factor = fpx / origW (matches sharp_post's d_factor)
    float disparityFactor = (jOrigW > 0) ? ((float)jFpx / (float)jOrigW) : 0.0f;
    std::string dfPath = workDir + "/disparity_factor.raw";
    {
        std::ofstream f(dfPath, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&disparityFactor), sizeof(float));
    }

    // 显式输入: image + disparity_factor
    // Explicit inputs: image + disparity_factor
    // 其余输入 (Seg A 边张量 + disparity) 通过缓存或目录搜索自动匹配
    // The rest (Seg A edge tensors + disparity) are matched via the cache or directory search
    std::map<std::string, std::string> explicitInputs = {
        {"image", workDir + "/image.raw"},
        {"disparity_factor", dfPath},
    };
    std::string outDir = workDir + "/rest_c_out";

    LOGI("runRestSegC entry RSS=%zuKB", selfVmRSS_kB());
    if (!runRestInference(g_restCRuntime.get(), explicitInputs,
                          {workDir + "/rest_a_out", workDir + "/rest_b_out"},
                           outDir, 7, "Gaussian Delta")) {
        return JNI_FALSE;
    }

    // 查找 delta 输出并复制到 workDir/delta.raw (供 Post 使用)
    // Find the delta output and copy it to workDir/delta.raw (used by Post)
    const auto& cOutputs = g_restCRuntime->getOutputInfos();
    bool deltaFound = false;
    for (const auto& out : cOutputs) {
        if (out.name == "delta" || out.name.find("delta") != std::string::npos) {
            std::string src = outDir + "/" + sanitizeTensorName(out.name) + ".raw";
            std::string dst = workDir + "/delta.raw";
            {
                std::ifstream srcF(src, std::ios::binary);
                std::ofstream dstF(dst, std::ios::binary);
                if (!srcF || !dstF) {
                    LOGE("delta copy failed");
                    cbError(7, "err_delta_copy_failed");
                    return JNI_FALSE;
                }
                dstF << srcF.rdbuf();
            }
            deltaFound = true;
            break;
        }
    }
    if (!deltaFound) {
        LOGE("delta output tensor not found");
        cbError(7, "err_delta_output_missing");
        return JNI_FALSE;
    }

    // 清理 REST 缓存 (Pipeline 推理已完成)
    // Clear the REST cache (the pipeline inference is complete)
    g_restCache.clear();
    g_restCacheBytes = 0;

    LOGI("runRestSegC exit RSS=%zuKB", selfVmRSS_kB());
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_runPost(JNIEnv* env, jobject thiz,
                                            jstring jWorkDir, jfloat jFpx, jint jOrigW, jint jOrigH,
                                            jstring jOutPlyPath) {
    std::string workDir = jstrToString(env, jWorkDir);
    std::string outPlyPath = jstrToString(env, jOutPlyPath);

    LOGI("runPost entry RSS=%zuKB", selfVmRSS_kB());
    int ret = sharp_post(workDir.c_str(), (float)jFpx, (int)jOrigW, (int)jOrigH, outPlyPath.c_str());
    LOGI("runPost exit RSS=%zuKB", selfVmRSS_kB());
    if (ret != 0) {
        LOGE("post failed: %d", ret);
        cbError(8, "err_pointcloud_failed");
        return JNI_FALSE;
    }

    return JNI_TRUE;
}

// ====== 模型内存管理 ======
// ====== Model memory management ======

JNIEXPORT void JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_freeModel(JNIEnv* env, jobject thiz, jstring jModelType) {
    std::string modelType = jstrToString(env, jModelType);
    auto& runtime = runtimeFor(modelType);
    if (runtime) {
        LOGI("freeModel before RSS=%zuKB", selfVmRSS_kB());
        runtime->freeGraph();
        LOGI("freed %s model memory (context+graph, runtime kept), RSS=%zuKB",
             modelType.c_str(), selfVmRSS_kB());
    }
}

JNIEXPORT void JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_clearRestCache(JNIEnv* env, jobject thiz) {
    size_t n = g_restCache.size();
    g_restCache.clear();
    g_restCacheBytes = 0;
    LOGI("REST cache cleared (%zu tensors)", n);
}

// ====== 进度回调设置 ======
// ====== Progress callback setup ======

JNIEXPORT void JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_setProgressCallback(JNIEnv* env, jobject thiz, jobject jCallback) {
    if (g_callbackObj) {
        env->DeleteGlobalRef(g_callbackObj);
        g_callbackObj = nullptr;
    }
    if (jCallback) {
        g_callbackObj = env->NewGlobalRef(jCallback);
        if (checkJniException(env) || !g_callbackObj) {
            g_callbackObj = nullptr;
            return;
        }
        jclass cls = env->GetObjectClass(g_callbackObj);
        if (checkJniException(env) || !cls) {
            env->DeleteGlobalRef(g_callbackObj);
            g_callbackObj = nullptr;
            return;
        }
        g_onStageStart   = env->GetMethodID(cls, "onStageStart",   "(ILjava/lang/String;)V");
        g_onProgress     = env->GetMethodID(cls, "onProgress",     "(IIIJLjava/lang/String;)V");
        g_onStageComplete= env->GetMethodID(cls, "onStageComplete","(ILjava/lang/String;J)V");
        g_onLog          = env->GetMethodID(cls, "onLog",          "(ILjava/lang/String;)V");
        g_onError        = env->GetMethodID(cls, "onError",        "(ILjava/lang/String;)V");
        if (checkJniException(env) || !g_onStageStart || !g_onProgress || !g_onStageComplete || !g_onLog || !g_onError) {
            env->DeleteGlobalRef(g_callbackObj);
            g_callbackObj = nullptr;
            g_onStageStart = g_onProgress = g_onStageComplete = g_onLog = g_onError = nullptr;
        }
        env->DeleteLocalRef(cls);
        LOGI("progress callback set");
    }
}

// ====== 模型文件验证 ======
// ====== Model file validation ======

// 验证模型文件完整性: 返回 null 表示合法, 否则返回错误消息
// Validates model file integrity: returns null if valid, an error message otherwise
// 检查文件存在性、大小合理性、基础格式头
// Checks file existence, reasonable size, and basic format header
JNIEXPORT jstring JNICALL
Java_com_sharp_qnn_pipeline_QnnJni_validateModelFile(JNIEnv* env, jobject thiz,
                                                      jstring jPath, jstring jFormat) {
    std::string path = jstrToString(env, jPath);
    std::string format = jstrToString(env, jFormat);

    // 检查文件是否存在且可读
    // Check that the file exists and is readable
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return env->NewStringUTF("File not found or not readable");
    }

    // 获取文件大小
    // Get the file size
    f.seekg(0, std::ios::end);
    size_t size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);

    if (size == 0) {
        return env->NewStringUTF("File is empty (0 bytes)");
    }
    if (size > 1024 * 1024 * 1024) { // > 1GB 不合理 / > 1GB is unreasonable
        return env->NewStringUTF("File too large (> 1GB)");
    }

    // 读取文件头 16 字节用于格式检查
    // Read the first 16 bytes for format checking
    char header[16] = {};
    f.read(header, sizeof(header));
    size_t readBytes = (size_t)f.gcount();
    f.close();

    if (format == "bin") {
        // QNN context binary: 至少 16 字节, 版本号字段应非零
        // QNN context binary: at least 16 bytes, the version field should be non-zero
        if (size < 16) {
            return env->NewStringUTF("Context binary too small (< 16 bytes)");
        }
        uint64_t version = 0;
        memcpy(&version, header, sizeof(version));
        if (version == 0) {
            return env->NewStringUTF("Invalid context binary: version field is zero");
        }
        LOGI("validateModelFile: bin OK, size=%zu bytes, version=0x%llx", size, (unsigned long long)version);
    } else if (format == "dlc") {
        // DLC: FlatBuffers 格式, 前 4 字节为 root table 偏移 (小端)
        // DLC: FlatBuffers format, the first 4 bytes are the root table offset (little-endian)
        if (size < 8) {
            return env->NewStringUTF("DLC file too small (< 8 bytes)");
        }
        uint32_t rootOffset = 0;
        memcpy(&rootOffset, header, sizeof(rootOffset));
        // FlatBuffers root offset 不应超过文件大小
        // FlatBuffers root offset should not exceed the file size
        if (rootOffset >= size) {
            return env->NewStringUTF("Invalid DLC: root table offset exceeds file size");
        }
        LOGI("validateModelFile: DLC OK, size=%zu bytes, rootOffset=%u", size, rootOffset);
    } else {
        return env->NewStringUTF("Unknown model format");
    }

    return nullptr; // 合法 / valid
}

} // extern "C"